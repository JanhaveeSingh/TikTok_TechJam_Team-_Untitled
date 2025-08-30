import time
from typing import Any, Optional, Tuple
from dotenv import load_dotenv
import os

# Add rate limiting to prevent API quota issues
_last_api_call = 0
_api_call_interval = 3  # Minimum 3 seconds between API calls for free tier

# Load environment variables from .env file
load_dotenv()
import config

from langchain.prompts import ChatPromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from pydantic import SecretStr
from data.State import DeploymentState, ElementMatch
from data.graph_db import Neo4jDatabase
from data.vector_db import VectorStore
from tool.img_tool import *
from tool.screen_content import *
from unified_parser import parse_ui_elements, get_parser_status
from dynamic_completion import wait_for_action_completion, check_ui_readiness

os.environ["LANGCHAIN_TRACING_V2"] = config.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = config.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "DeploymentExecution"

model = ChatGoogleGenerativeAI(
    google_api_key=config.GEMINI_API_KEY,
    model=config.LLM_MODEL,
    max_output_tokens=config.LLM_MAX_TOKEN,
    # Add other Gemini-specific parameters if needed
)

URI = config.Neo4j_URI
AUTH = config.Neo4j_AUTH



# Try alternative URI schemes for Neo4j Aura if routing fails
alternative_uris = []
if URI:
    alternative_uris.append(URI)  # Original URI
    if URI.startswith("neo4j+s://"):
        # Try neo4j+ssc:// (self-signed certificate) for SSL issues
        ssc_uri = URI.replace("neo4j+s://", "neo4j+ssc://")
        alternative_uris.append(ssc_uri)
        # Try bolt+ssc:// instead of bolt+s:// for SSL issues
        bolt_ssc_uri = URI.replace("neo4j+s://", "bolt+ssc://")
        alternative_uris.append(bolt_ssc_uri)

db = None
for attempt, uri in enumerate(alternative_uris, 1):
    try:
        print(f"[Deployment] Attempt {attempt}: Trying URI {uri}")
        db = Neo4jDatabase(uri, AUTH)
        print(f"[Deployment] Successfully connected to Neo4j using {uri}")
        break
    except Exception as e:
        print(f"[Deployment] Attempt {attempt} failed with URI {uri}: {e}")
        if attempt == len(alternative_uris):
            print("[Deployment] All connection attempts failed.")
            raise

vector_db = VectorStore(api_key=config.PINECONE_API_KEY)

# Working UI Agent functionality has been integrated into dynamic_completion.py
working_ui_agent = None


def rate_limit_api_call():
    """Ensure minimum interval between API calls to avoid quota issues."""
    global _last_api_call, _api_call_interval
    current_time = time.time()
    time_since_last = current_time - _last_api_call
    
    if time_since_last < _api_call_interval:
        sleep_time = _api_call_interval - time_since_last
        print(f"⏳ Rate limiting: waiting {sleep_time:.1f}s to avoid API quota...")
        time.sleep(sleep_time)
    
    _last_api_call = time.time()


def create_execution_state(device: str) -> Dict[str, Any]:
    """
    Create initial execution state

    Args:
        device: Device ID

    Returns:
        Dictionary containing initial state
    """
    from data.State import create_deployment_state

    state = create_deployment_state(
        task="",
        device=device,
    )

    return state


def match_task_to_action(
    state: Dict[str, Any], task: str
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Match user task with high-level action nodes

    Args:
        state: Execution state
        task: User input task description

    Returns:
        (whether match successful, matched high-level action node)
    """
    print(f"Matching task: {task}")

    # 1. Get all high-level action nodes from database
    high_level_actions = db.get_all_high_level_actions()

    if not high_level_actions:
        print("❌ No high-level action nodes found")
        return False, None

    print(f"Found {len(high_level_actions)} high-level action nodes")

    if len(high_level_actions) == 0:
        return False, None

    # 2. Create task matching prompt
    task_match_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an AI assistant specialized in matching user tasks with predefined high-level actions.
You need to analyze the user's task description and determine if it matches any predefined high-level actions.
If you find a matching high-level action, return the complete information of that action. If no match is found, clearly indicate no match.
Only consider it a match when the matching degree is high (above 0.7).""",
            ),
            (
                "human",
                """User task: {task}

Available high-level actions:
{actions_json}

Please determine if the user task matches any high-level action.
If matched successfully, return the complete information of the best matching action (keeping the original JSON format) with "MATCHED: " prefix.
If no match is found, return "NO_MATCH" with a brief explanation.
""",
            ),
        ]
    )

    # 3. Prepare JSON string of high-level actions
    actions_json = json.dumps(high_level_actions, ensure_ascii=False, indent=2)

    # 4. Call LLM for matching
    try:
        # Prepare input
        match_input = {"task": task, "actions_json": actions_json}

        # Use simple string output parser
        match_chain = task_match_prompt | model | StrOutputParser()

        # Execute matching
        result = match_chain.invoke(match_input)

        # Parse results
        if result.startswith("MATCHED:"):
            # Extract matched action information
            action_json_str = result[len("MATCHED:") :].strip()
            try:
                matched_action = json.loads(action_json_str)
                print(
                    f"✓ Found matching high-level action: {matched_action.get('name', 'Unknown')} (ID: {matched_action.get('action_id', 'Unknown')})"
                )
                return True, matched_action
            except json.JSONDecodeError:
                print(f"❌ Cannot parse matching result: {action_json_str}")
                return False, None
        elif result.startswith("NO_MATCH"):
            reason = result[len("NO_MATCH") :].strip()
            print(f"❌ No matching high-level action found")
            print(f"  Reason: {reason}")
            return False, None
        else:
            print(f"❌ Unrecognized matching result: {result}")
            return False, None

    except Exception as e:
        print(f"❌ Error during task matching: {str(e)}")
        return False, None


def capture_and_parse_screen(state: DeploymentState) -> DeploymentState:
    """
    Capture current screen and parse elements, update state

    Args:
        state: Deployment state

    Returns:
        Updated deployment state
    """
    try:
        # 1. Take screenshot
        screenshot_path = take_screenshot.invoke(
            {
                "device": state["device"],
                "app_name": "deployment",
                "step": state["current_step"],
            }
        )

        if not screenshot_path or not os.path.exists(screenshot_path):
            print("❌ Screenshot failed")
            return state

        # 2. Parse screen elements using unified parser system
        try:
            # Use unified parser for better performance and compatibility
            parser_result = parse_ui_elements(screenshot_path)
            
            if parser_result["status"] != "success":
                print(f"❌ Unified parser failed: {parser_result}")
                # Fallback to original screen_element
                screen_result = screen_element.invoke({"image_path": screenshot_path})
                if "error" in screen_result:
                    print(f"❌ Screen element parsing failed: {screen_result['error']}")
                    return state
            else:
                print(f"✅ Parsed with {parser_result['parser_used']} parser in {parser_result['processing_time']:.2f}s")
                # Convert unified parser result to expected format
                screen_result = {
                    "parsed_content_json_path": screenshot_path.replace('.png', '_parsed.json'),
                    "parsed_content": parser_result["parsed_content"]
                }
                
                # Save parsed content to JSON file for compatibility
                with open(screen_result["parsed_content_json_path"], 'w', encoding='utf-8') as f:
                    json.dump(parser_result["parsed_content"], f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"⚠️ Unified parser error: {e}, falling back to original parser")
            # Fallback to original screen_element
            screen_result = screen_element.invoke({"image_path": screenshot_path})
            if "error" in screen_result:
                print(f"❌ Screen element parsing failed: {screen_result['error']}")
                return state

        # 3. Update current page information
        state["current_page"]["screenshot"] = screenshot_path
        state["current_page"]["elements_json"] = screen_result[
            "parsed_content_json_path"
        ]

        # 4. Load element data
        with open(
            screen_result["parsed_content_json_path"], "r", encoding="utf-8"
        ) as f:
            state["current_page"]["elements_data"] = json.load(f)

        print(
            f"✓ Successfully parsed current screen, detected {len(state['current_page']['elements_data'])} UI elements"
        )
        return state

    except Exception as e:
        print(f"❌ Error capturing and parsing screen: {str(e)}")
        return state


def match_screen_elements(
    state: DeploymentState, action_sequence: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Match current screen elements with elements in high-level action nodes using visual embedding comparison

    Args:
        state: Deployment state
        action_sequence: Element sequence in high-level action nodes

    Returns:
        List of matching results, including screen element ID and matching score
    """
    if not state["current_page"]["elements_data"]:
        print("❌ Current screen element data is empty")
        return []

    # Get current step information
    current_step_idx = state["current_step"]
    if current_step_idx >= len(action_sequence):
        print(f"⚠️ Current step {current_step_idx} exceeds action sequence range")
        return []

    current_action = action_sequence[current_step_idx]

    # Get element information
    element_id = current_action.get("element_id")
    if not element_id:
        print("⚠️ No element ID specified in current step")
        return []

    # Get template element from database - using correct method name
    db_element = db.get_action_by_id(element_id)
    if not db_element:
        print(f"⚠️ Element with ID {element_id} not found")
        # Try to get from another type
        db_element = db.get_element_by_id(element_id)
        if not db_element:
            print(f"⚠️ Action with ID {element_id} also not found")
            return []

    # If retrieved node is an Action node, ensure it contains necessary visual information
    # Otherwise fall back to semantic matching
    if "action_id" in db_element and not any(
        key in db_element for key in ["visual_embedding", "screenshot_path"]
    ):
        print(
            f"⚠️ Retrieved node is an Action node but lacks visual information, falling back to semantic matching"
        )
        return fallback_to_semantic_match(state, action_sequence)

    # Check if visual embedding exists
    template_embedding = None
    if "visual_embedding" in db_element and db_element["visual_embedding"]:
        template_embedding = db_element["visual_embedding"]
    else:
        print("⚠️ Template element has no visual embedding, trying to extract features")
        # Try to get element screenshot or extract features using bounding box information
        if "screenshot_path" in db_element and db_element["screenshot_path"]:
            try:
                # Extract template element features
                template_embedding = extract_features(
                    db_element["screenshot_path"], "resnet50"
                )["features"]
            except Exception as e:
                print(f"❌ Cannot extract template element features: {str(e)}")
                return fallback_to_semantic_match(state, action_sequence)
        else:
            # No visual embedding, fall back to semantic matching
            print(
                "⚠️ Cannot get template element visual features, falling back to semantic matching"
            )
            return fallback_to_semantic_match(state, action_sequence)

    # Process current screen elements
    screen_elements = state["current_page"]["elements_data"]
    screenshot_path = state["current_page"]["screenshot"]
    elements_json_path = state["current_page"]["elements_json"]

    try:
        # Get visual embeddings for all elements on current screen
        from tool.img_tool import elements_img, extract_features

        # Extract features for all screen elements
        element_embeddings = []
        for idx, element in enumerate(screen_elements):
            try:
                # Use element_img to get element image
                element_img_stream = elements_img(
                    screenshot_path, json.dumps(screen_elements), element.get("ID", idx)
                )
                # Extract features
                element_feature = extract_features(element_img_stream, "resnet50")
                element_embeddings.append((idx, element_feature["features"]))
            except Exception as e:
                print(f"⚠️ Cannot extract features for element {idx}: {str(e)}")
                continue

        if not element_embeddings:
            print("❌ Failed to extract features for any screen elements")
            return fallback_to_semantic_match(state, action_sequence)

        # Calculate similarity and sort
        import numpy as np

        matches = []
        for idx, embedding in element_embeddings:
            # Calculate cosine similarity
            template_vec = np.array(template_embedding).flatten()
            element_vec = np.array(embedding).flatten()

            # Normalize vectors
            template_norm = np.linalg.norm(template_vec)
            element_norm = np.linalg.norm(element_vec)

            if template_norm == 0 or element_norm == 0:
                similarity = 0
            else:
                similarity = np.dot(template_vec, element_vec) / (
                    template_norm * element_norm
                )

            # Convert similarity to match score
            match_score = float(similarity)

            if match_score >= 0.6:  # Matching threshold
                matches.append(
                    {
                        "element_id": element_id,
                        "match_score": match_score,
                        "screen_element_id": idx,
                        "action_type": current_action.get("atomic_action", "tap"),
                        "parameters": current_action.get("action_params", {}),
                    }
                )

        # Sort by similarity
        matches.sort(key=lambda x: x["match_score"], reverse=True)

        if matches:
            best_match = matches[0]
            print(
                f"✓ Found matching screen element ID: {best_match['screen_element_id']}"
            )
            print(f"  Match score: {best_match['match_score']}")
            print(f"  Action type: {best_match['action_type']}")
            return matches
        else:
            print("❌ No matching screen element found")
            return []

    except Exception as e:
        print(f"❌ Error during visual matching: {str(e)}")
        # Fall back to semantic matching on error
        return fallback_to_semantic_match(state, action_sequence)


def fallback_to_semantic_match(
    state: DeploymentState, action_sequence: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Fallback to semantic matching when visual matching fails
    """
    print("🔄 Falling back to semantic matching...")

    # Prepare template element features
    template_elements = []
    current_step_idx = state["current_step"]
    if current_step_idx >= len(action_sequence):
        return []

    step_info = action_sequence[current_step_idx]

    # Get element information
    element_id = step_info.get("element_id")
    if not element_id:
        return []

    # Get element information from database - using correct method name
    db_element = db.get_action_by_id(element_id)
    if not db_element:
        print(f"⚠️ Element with ID {element_id} not found")
        # Try to get from another type
        db_element = db.get_element_by_id(element_id)
        if not db_element:
            print(f"⚠️ Action with ID {element_id} also not found")
            return []

    template_elements.append({"db_element": db_element, "step_info": step_info})

    # If no elements to match, return empty list
    if not template_elements:
        return []

    current_template = template_elements[0]

    # Prepare matching prompt
    element_match_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an AI assistant specialized in matching UI elements. You need to analyze template element descriptions and current screen elements to find the best match. Your answer must be in JSON format, including the matching results.""",
            ),
            (
                "human",
                """Template element description: 
{template_element}

Current screen elements:
{screen_elements}

Please find the screen element that best matches the template element, and return in the following JSON format:
{
  "element_id": "template element ID",
  "match_score": matching score (0-1),
  "screen_element_id": screen element ID,
  "action_type": "atomic action type (tap/text/swipe etc.)",
  "parameters": {action parameters object}
}

If no element with matching score above 0.6 is found, set match_score to 0 and screen_element_id to -1.
""",
            ),
        ]
    )

    # Parse template element information
    current_db_element = current_template["db_element"]

    # Determine correct ID field
    element_id_field = (
        "element_id" if "element_id" in current_db_element else "action_id"
    )
    template_element_desc = (
        f"ID: {current_db_element.get(element_id_field, 'unknown')}\n"
    )

    if "description" in current_db_element and current_db_element["description"]:
        template_element_desc += f"Description: {current_db_element['description']}\n"
    elif "name" in current_db_element and current_db_element["name"]:
        template_element_desc += f"Name: {current_db_element['name']}\n"

    # Check position information, supporting different field names
    bbox_field = None
    for field in ["bounding_box", "bbox", "position"]:
        if field in current_db_element and current_db_element[field]:
            bbox_field = field
            break

    if bbox_field:
        bbox = current_db_element[bbox_field]
        if isinstance(bbox, list) and len(bbox) >= 4:
            template_element_desc += f"Position: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]\n"
        elif isinstance(bbox, str):
            template_element_desc += f"Position: {bbox}\n"

    # Add action information
    template_element_desc += (
        f"Action type: {current_template['step_info'].get('atomic_action', 'tap')}\n"
    )

    if (
        "action_params" in current_template["step_info"]
        and current_template["step_info"]["action_params"]
    ):
        params = current_template["step_info"]["action_params"]
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except:
                pass

        if isinstance(params, dict):
            template_element_desc += "Action parameters:\n"
            for k, v in params.items():
                template_element_desc += f"  {k}: {v}\n"

    # Parse screen elements information
    screen_elements_desc = ""
    for i, element in enumerate(state["current_page"]["elements_data"]):
        screen_elements_desc += f"Element {i} (ID: {element.get('ID', i)}):\n"
        if "type" in element:
            screen_elements_desc += f"  Type: {element['type']}\n"
        if "content" in element:
            screen_elements_desc += f"  Content: {element['content']}\n"
        if "bbox" in element:
            bbox = element["bbox"]
            screen_elements_desc += f"  Position: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]\n"
        screen_elements_desc += "\n"

    # Call LLM for matching
    try:
        # Prepare input
        match_input = {
            "template_element": template_element_desc,
            "screen_elements": screen_elements_desc,
        }

        # Create parser
        parser = JsonOutputParser(pydantic_object=ElementMatch)

        # Build chain
        match_chain = element_match_prompt | model | parser

        # Execute matching
        match_result = match_chain.invoke(match_input)

        # Check matching result
        if match_result.match_score >= 0.6 and match_result.screen_element_id >= 0:
            print(
                f"✓ Found matching screen element ID: {match_result.screen_element_id}"
            )
            print(f"  Match score: {match_result.match_score}")
            print(f"  Action type: {match_result.action_type}")
            return [match_result.dict()]
        else:
            print(f"❌ No matching screen element found")
            return []

    except Exception as e:
        print(f"❌ Error during element matching: {str(e)}")
        return []


def execute_element_action(
    state: DeploymentState, element_match: Dict[str, Any]
) -> bool:
    """
    Execute screen element action

    Args:
        state: Execution state
        element_match: Element matching result

    Returns:
        Whether the operation was successful
    """
    try:
        if not element_match:
            return False

        # Get action type and parameters
        action_type = element_match.get("action_type", "tap")
        parameters = element_match.get("parameters", {})
        screen_element_id = element_match.get("screen_element_id", -1)

        if screen_element_id < 0 or screen_element_id >= len(
            state["current_page"]["elements_data"]
        ):
            print(f"❌ Invalid screen element ID: {screen_element_id}")
            return False

        # Get element position
        element = state["current_page"]["elements_data"][screen_element_id]
        bbox = element.get("bbox", [0, 0, 0, 0])

        # Get device dimensions dynamically
        device_size = get_device_size.invoke(state["device"])
        if isinstance(device_size, str):
            if "x" in device_size:
                try:
                    width, height = map(int, device_size.split("x"))
                except:
                    width, height = 1080, 2424  # Default fallback
            else:
                width, height = 1080, 2424
        else:
            width = device_size.get("width", 1080)
            height = device_size.get("height", 2424)
        
        # Calculate center point with bounds checking
        center_x = max(10, min(width - 10, int((bbox[0] + bbox[2]) / 2 * width)))
        center_y = max(10, min(height - 10, int((bbox[1] + bbox[3]) / 2 * height)))
        
        print(f"🎯 Calculated tap coordinates: ({center_x}, {center_y}) for element with bbox {bbox}")
        print(f"📱 Device dimensions: {width}x{height}, coordinates are absolute pixels")

        # Prepare action parameters
        action_params = {
            "device": state["device"],
            "action": action_type,
            "x": center_x,
            "y": center_y,
        }

        # Add specific parameters based on action type
        if action_type == "text":
            action_params["input_str"] = parameters.get("text", "")
        elif action_type == "long_press":
            action_params["duration"] = parameters.get("duration", 1000)
        elif action_type == "swipe":
            action_params["direction"] = parameters.get("direction", "up")
            action_params["dist"] = parameters.get("distance", "medium")

        # Execute action
        print(f"Executing action: {action_type} at position ({center_x}, {center_y})")
        result = screen_action.invoke(action_params)

        # Parse operation result
        if isinstance(result, str):
            try:
                result_json = json.loads(result)
                if result_json.get("status") == "success":
                    print(f"✓ Action executed successfully")
                    
                    # Add dynamic completion detection after successful action
                    try:
                        def screenshot_func():
                            """Take screenshot for completion detection"""
                            return take_screenshot.invoke({
                                "device": state["device"],
                                "app_name": "completion_check",
                                "step": state.get("current_step", 0)
                            })
                        
                        # Wait for UI to complete the action
                        print("🔍 Waiting for UI action to complete...")
                        completion_result = wait_for_action_completion(
                            screenshot_func, 
                            action_type, 
                            max_wait=3.0
                        )
                        
                        if completion_result.get("completed", False):
                            wait_time = completion_result.get("wait_time", 0)
                            print(f"✅ Action completion detected in {wait_time:.2f}s")
                        else:
                            print("⚠️ Action completion timeout - continuing anyway")
                            
                    except Exception as e:
                        print(f"⚠️ Completion detection failed: {e} - continuing anyway")
                    
                    return True
                else:
                    print(
                        f"❌ Action execution failed: {result_json.get('message', 'Unknown error')}"
                    )
                    return False
            except:
                print(f"❌ Cannot parse operation result: {result}")
                return False
        else:
            print(f"❌ Operation returned non-string result")
            return False

    except Exception as e:
        print(f"❌ Error executing element action: {str(e)}")
        return False


def fallback_to_react(state: DeploymentState) -> DeploymentState:
    """
    Fall back to React mode when template execution fails

    Args:
        state: Execution state

    Returns:
        Updated execution state
    """
    print("🔄 Falling back to React mode execution...")
    task = state["task"]

    # Create action_agent for page operation decisions
    action_agent = create_react_agent(model, [screen_action])

    # Initialize React mode
    if not state["messages"]:
        # Set system prompt
        system_message = SystemMessage(
            content="""You are an intelligent smartphone operation assistant who will help users complete tasks on mobile devices.
You can help users by observing the screen and performing various operations (clicking, typing text, swiping, etc.).
Analyze the current screen content, determine the best next action, and use the appropriate tools to execute it.
Each step of the operation should move toward completing the user's goal task."""
        )

        state["messages"].append(system_message)

        # Add user task
        user_message = HumanMessage(
            content=f"I need to complete the following task on a mobile device: {task}"
        )
        state["messages"].append(user_message)

    # Capture current screen
    state = capture_and_parse_screen(state)
    if not state["current_page"]["screenshot"]:
        state["execution_status"] = "error"
        print("Unable to capture or parse screen")
        return state
    
    # Enhanced element detection for specific tasks
    if "tiktok" in task.lower() or "open app" in task.lower():
        # Look for app icons specifically in parsed elements
        parsed_elements = state["current_page"].get("elements_data", [])
        
        # Get device dimensions dynamically
        device_size = get_device_size.invoke(state["device"])
        if isinstance(device_size, str):
            # Parse device size string or use defaults
            if "x" in device_size:
                try:
                    width, height = map(int, device_size.split("x"))
                except:
                    width, height = 1080, 2424  # Default to Pixel 9a
            else:
                width, height = 1080, 2424
        else:
            width = device_size.get("width", 1080)
            height = device_size.get("height", 2424)
        
        print(f"📱 Device dimensions: {width}x{height}")
        
        # Debug: Show what elements are detected
        debug_screen_elements(parsed_elements, task)
        
        if parsed_elements:
            app_icon = find_app_icon(parsed_elements, "tiktok", width, height)
            if app_icon:
                print(f"🎯 Found TikTok app icon at ({app_icon['x']}, {app_icon['y']})")
                # Execute tap on the app icon using safe function
                if safe_tap_coordinates(state["device"], app_icon["x"], app_icon["y"], width, height):
                    print("✅ Successfully tapped TikTok app icon")
                    state["execution_status"] = "completed"
                    state["completed"] = True
                    return state
                else:
                    print("⚠️ Failed to tap app icon - continuing with React mode")
            
            # Also look for common permission buttons
            permission_button = find_ui_element_by_text(parsed_elements, "while using the app", width, height)
            if permission_button:
                print(f"🎯 Found permission button at ({permission_button['x']}, {permission_button['y']})")
                if safe_tap_coordinates(state["device"], permission_button["x"], permission_button["y"], width, height):
                    print("✅ Successfully tapped permission button")
                    # Don't mark as completed yet, continue to next step
                else:
                    print("⚠️ Failed to tap permission button - continuing with React mode")

    # Prepare screen information
    screenshot_path = state["current_page"]["screenshot"]
    elements_json_path = state["current_page"]["elements_json"]
    device = state["device"]
    device_size = get_device_size.invoke(state["device"])

    # Load screenshot as base64
    with open(screenshot_path, "rb") as f:
        image_data = f.read()
        image_data_base64 = base64.b64encode(image_data).decode("utf-8")

    # Load element JSON data
    with open(elements_json_path, "r", encoding="utf-8") as f:
        elements_data = json.load(f)
    
    # Fix coordinates dynamically based on device size
    for element in elements_data:
        if "bbox" in element and len(element["bbox"]) == 4:
            bbox = element["bbox"]
            print(f"🔍 Converting element bbox: {bbox}")
            
            # Check if bbox values are already absolute pixels or relative (0-1)
            if max(bbox) > 1.0:  # Likely absolute pixels
                print(f"   Bbox appears to be absolute pixels, using as-is")
                x = int(bbox[0])
                y = int(bbox[1])
                element_width = int(bbox[2] - bbox[0]) if bbox[2] > bbox[0] else int(bbox[2])
                element_height = int(bbox[3] - bbox[1]) if bbox[3] > bbox[1] else int(bbox[3])
            else:  # Relative coordinates (0-1) - always treat as corner-based [x1, y1, x2, y2]
                print(f"   Bbox appears to be relative coordinates, converting to absolute")
                x = int(bbox[0] * width)
                y = int(bbox[1] * height)
                element_width = int((bbox[2] - bbox[0]) * width)
                element_height = int((bbox[3] - bbox[1]) * height)
            
            # Update element with absolute coordinates
            element["x"] = x
            element["y"] = y
            element["width"] = element_width
            element["height"] = element_height
            element["center_x"] = x + (element_width // 2)
            element["center_y"] = y + (element_height // 2)
            
            print(f"   Converted to: x={x}, y={y}, w={element_width}, h={element_height}")
            print(f"   Center: ({element['center_x']}, {element['center_y']})")

    elements_text = json.dumps(elements_data, ensure_ascii=False, indent=2)

    # Build messages
    messages = [
        SystemMessage(
            content=f"""Below is the current page information and user intent. Please analyze comprehensively and recommend the next reasonable action (please complete only one step),
and complete it by calling tools. All tool calls must pass in device to specify the operating device. Only execute one tool call.

IMPORTANT: The device has screen dimensions {width}x{height}. All coordinates in the elements data have been converted to absolute pixel values.
Use the center_x and center_y values for precise tapping, or calculate the center of any element using x + (width/2) and y + (height/2).

STRATEGY: 
1. For opening apps: Look for app icons by name or icon class
2. For permissions: Look for buttons with text like "While using the app", "Allow", "OK"
3. For navigation: Use precise coordinates from center_x/center_y
4. Always validate coordinates are within 0-{width} (width) and 0-{height} (height)"""
        ),
        HumanMessage(
            content=f"The current device is: {device}, the device screen size is {device_size}. The user's current task intent is: {task}"
        ),
        HumanMessage(
            content="Below is the current page's parsed JSON data (where bbox is a relative value, please convert to actual operation position based on screen size):\n"
            + elements_text
        ),
        HumanMessage(
            content=[
                {"type": "text", "text": "Below is the screenshot data:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data_base64}"},
                },
            ],
        ),
    ]

    # Add these messages to state
    state["messages"].extend(messages)

    # Check if we should stop due to repeated actions or API limits
    if len(state["history"]) >= 5:
        recent_actions = [step.get("action", "") for step in state["history"][-3:]]
        if len(set(recent_actions)) <= 2:  # Only 1-2 unique actions in last 3 steps
            print("🛑 Stopping due to repeated actions - likely task completed or stuck")
            state["execution_status"] = "completed"
            state["completed"] = True
            return state
    
    # Enhanced guidance for React mode
    print("🎯 React mode: Use center_x and center_y coordinates for precise tapping")
    print("📱 Device: Pixel 9a (1080x2424) - coordinates are absolute pixels")
    
    # Debug: Show detected elements
    if parsed_elements:
        print(f"🔍 Detected {len(parsed_elements)} UI elements:")
        for i, elem in enumerate(parsed_elements[:5]):  # Show first 5 elements
            text = elem.get("text", "No text")[:30]  # Truncate long text
            x, y = elem.get("x", 0), elem.get("y", 0)
            print(f"  {i}: '{text}' at ({x}, {y})")
        if len(parsed_elements) > 5:
            print(f"  ... and {len(parsed_elements) - 5} more elements")

    # Call action_agent for decision making and action execution
    try:
        rate_limit_api_call()  # Prevent API quota issues
        action_result = action_agent.invoke({"messages": state["messages"][-4:]})
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "429" in error_msg:
            print("🛑 Stopping due to API quota limits")
            state["execution_status"] = "completed"
            state["completed"] = True
            return state
        else:
            print(f"❌ Action agent failed: {error_msg}")
            state["execution_status"] = "error"
            return state

    # Parse results
    final_messages = action_result.get("messages", [])
    if final_messages:
        # Add AI reply to message history
        ai_message = final_messages[-1]
        state["messages"].append(ai_message)

        # Extract recommended action from final_message
        recommended_action = ai_message.content.strip()

        # Update execution status
        state["current_step"] += 1
        state["history"].append(
            {
                "step": state["current_step"],
                "screenshot": screenshot_path,
                "elements_json": elements_json_path,
                "action": "react_mode",
                "recommended_action": recommended_action,
                "status": "success",
            }
        )

        state["execution_status"] = "success"
        print(f"✓ React mode execution successful: {recommended_action}")
        
        # For simple single-action tasks, mark as completed immediately
        task_lower = state["task"].lower()
        if any(word in task_lower for word in ["click", "tap", "press", "select"]):
            print("🎯 Simple action task completed - marking as finished")
            state["completed"] = True
    else:
        error_msg = "React mode execution failed: No messages returned"
        print(f"❌ {error_msg}")

        # Update execution status
        state["history"].append(
            {
                "step": state["current_step"],
                "screenshot": screenshot_path,
                "elements_json": elements_json_path,
                "action": "react_mode",
                "status": "error",
                "error": error_msg,
            }
        )

        state["execution_status"] = "error"

    return state


def find_app_icon(parsed_elements: List[Dict], app_name: str, device_width: int = 1080, device_height: int = 2424) -> Optional[Dict]:
    """Find app icon by name in parsed elements"""
    app_name_lower = app_name.lower()
    
    # First, try to find by exact text match
    for element in parsed_elements:
        element_text = element.get("text", "").lower()
        if app_name_lower in element_text:
            print(f"🎯 Found {app_name} by text: '{element.get('text', '')}'")
            return _get_element_coordinates(element, device_width, device_height)
    
    # Then try by class name containing icon/app
    for element in parsed_elements:
        element_class = element.get("class", "").lower()
        if "icon" in element_class or "app" in element_class:
            print(f"🎯 Found potential {app_name} by class: '{element.get('class', '')}'")
            return _get_element_coordinates(element, device_width, device_height)
    
    # Finally, look for any element that might be an app icon (has bbox and reasonable size)
    for element in parsed_elements:
        if "bbox" in element and len(element["bbox"]) == 4:
            bbox = element["bbox"]
            # Check if this looks like an app icon (reasonable size, not too small/large)
            width_ratio = bbox[2] - bbox[0]  # relative width
            height_ratio = bbox[3] - bbox[1]  # relative height
            
            # App icons are typically square-ish and not too small
            if (0.05 <= width_ratio <= 0.2 and 0.05 <= height_ratio <= 0.2 and 
                abs(width_ratio - height_ratio) < 0.05):  # roughly square
                
                print(f"🎯 Found potential app icon by size/shape: bbox={bbox}")
                return _get_element_coordinates(element, device_width, device_height)
    
    return None

def _get_element_coordinates(element: Dict, device_width: int, device_height: int) -> Optional[Dict]:
    """Helper function to extract coordinates from element"""
    print(f"🔍 Processing element: {element.get('text', 'No text')[:30]}")
    print(f"   Raw bbox: {element.get('bbox', 'No bbox')}")
    print(f"   Raw x/y: ({element.get('x', 'No x')}, {element.get('y', 'No y')})")
    
    # Get coordinates - use bbox if available, otherwise x/y
    if "bbox" in element and len(element["bbox"]) == 4:
        bbox = element["bbox"]
        print(f"   Processing bbox: {bbox}")
        
        # Check if bbox values are already absolute pixels (large numbers) or relative (0-1)
        if max(bbox) > 1.0:  # Likely absolute pixels
            print(f"   Bbox appears to be absolute pixels")
            x = int(bbox[0])
            y = int(bbox[1])
            width = int(bbox[2] - bbox[0]) if bbox[2] > bbox[0] else int(bbox[2])
            height = int(bbox[3] - bbox[1]) if bbox[3] > bbox[1] else int(bbox[3])
        else:  # Relative coordinates (0-1) - always treat as corner-based [x1, y1, x2, y2]
            print(f"   Bbox appears to be relative coordinates (corner-based)")
            x = int(bbox[0] * device_width)
            y = int(bbox[1] * device_height)
            width = int((bbox[2] - bbox[0]) * device_width)
            height = int((bbox[3] - bbox[1]) * device_height)
    else:
        # Use absolute coordinates if available
        x = element.get("x", 0)
        y = element.get("y", 0)
        width = element.get("width", 0)
        height = element.get("height", 0)
        print(f"   Using absolute coordinates: x={x}, y={y}, w={width}, h={height}")
    
    # Calculate center point
    center_x = x + (width // 2)
    center_y = y + (height // 2)
    
    print(f"   Calculated: x={x}, y={y}, w={width}, h={height}")
    print(f"   Center: ({center_x}, {center_y})")
    
    # Validate coordinates
    if (0 <= center_x <= device_width and 0 <= center_y <= device_height):
        print(f"✅ Valid coordinates: center=({center_x}, {center_y})")
        return {
            "x": center_x,
            "y": center_y,
            "width": width,
            "height": height,
            "text": element.get("text", ""),
            "class": element.get("class", ""),
            "bbox": element.get("bbox", [])
        }
    
    print(f"❌ Invalid coordinates: center=({center_x}, {center_y}) for device {device_width}x{device_height}")
    return None


def find_ui_element_by_text(parsed_elements: List[Dict], target_text: str, device_width: int = 1080, device_height: int = 2424) -> Optional[Dict]:
    """Find UI element by text content"""
    target_text_lower = target_text.lower()
    
    for element in parsed_elements:
        element_text = element.get("text", "").lower()
        if target_text_lower in element_text:
            # Get coordinates - use bbox if available, otherwise x/y
            if "bbox" in element and len(element["bbox"]) == 4:
                bbox = element["bbox"]
                # Check if bbox values are already absolute pixels or relative (0-1)
                if max(bbox) > 1.0:  # Likely absolute pixels
                    x = int(bbox[0])
                    y = int(bbox[1])
                    width = int(bbox[2] - bbox[0])
                    height = int(bbox[3] - bbox[1])
                else:  # Relative coordinates (0-1)
                    # IMPORTANT: Check if this is a center-based bbox or corner-based
                    if max(bbox) < 0.5:  # Likely center-based bbox
                        center_x_rel = bbox[0]
                        center_y_rel = bbox[1]
                        half_width_rel = bbox[2] / 2
                        half_height_rel = bbox[3] / 2
                        
                        # Convert to corner-based coordinates
                        x = int((center_x_rel - half_width_rel) * device_width)
                        y = int((center_y_rel - half_height_rel) * device_height)
                        width = int(bbox[2] * device_width)
                        height = int(bbox[3] * device_height)
                    else:  # Corner-based bbox
                        x = int(bbox[0] * device_width)
                        y = int(bbox[1] * device_height)
                        width = int((bbox[2] - bbox[0]) * device_width)
                        height = int((bbox[3] - bbox[1]) * device_height)
            else:
                # Use absolute coordinates if available
                x = element.get("x", 0)
                y = element.get("y", 0)
                width = element.get("width", 0)
                height = element.get("height", 0)
            
            # Calculate center point
            center_x = x + (width // 2)
            center_y = y + (height // 2)
            
            # Validate coordinates
            if (0 <= center_x <= device_width and 0 <= center_y <= device_height):
                print(f"🎯 Found text element '{target_text}': bbox={element.get('bbox', 'N/A')}, center=({center_x}, {center_y})")
                return {
                    "x": center_x,
                    "y": center_y,
                    "width": width,
                    "height": height,
                    "text": element.get("text", ""),
                    "class": element.get("class", ""),
                    "bbox": element.get("bbox", [])
                }
    
    return None


def validate_coordinates(x: int, y: int, device_width: int = 1080, device_height: int = 2424) -> bool:
    """Validate that coordinates are within device bounds"""
    print(f"🔍 Validating coordinates ({x}, {y}) for device {device_width}x{device_height}")
    
    if not (0 <= x <= device_width and 0 <= y <= device_height):
        print(f"⚠️ Invalid coordinates ({x}, {y}) for device {device_width}x{device_height}")
        return False
    
    print(f"✅ Coordinates are valid")
    return True


def safe_tap_coordinates(device: str, x: int, y: int, device_width: int = 1080, device_height: int = 2424) -> bool:
    """Safely tap at coordinates with validation"""
    print(f"🎯 Attempting to tap at coordinates: ({x}, {y})")
    print(f"   Device dimensions: {device_width}x{device_height}")
    
    if not validate_coordinates(x, y, device_width, device_height):
        return False
    
    try:
        result = screen_action.invoke({
            "device": device,
            "action": "tap",
            "x": x,
            "y": y
        })
        
        if "success" in str(result).lower():
            print(f"✅ Successfully tapped at ({x}, {y})")
            return True
        else:
            print(f"❌ Tap failed at ({x}, {y}): {result}")
            return False
            
    except Exception as e:
        print(f"❌ Error tapping at ({x}, {y}): {e}")
        return False


def debug_screen_elements(parsed_elements: List[Dict], task: str):
    """Debug function to show what elements are detected"""
    print(f"\n🔍 DEBUG: Screen analysis for task: '{task}'")
    print(f"📱 Total elements detected: {len(parsed_elements)}")
    
    if not parsed_elements:
        print("⚠️ No elements detected!")
        return
    
    # Show first few elements in detail
    print(f"🔍 First 5 elements detail:")
    for i, elem in enumerate(parsed_elements[:5]):
        print(f"  Element {i}:")
        print(f"    Text: '{elem.get('text', 'No text')}'")
        print(f"    Class: '{elem.get('class', 'No class')}'")
        print(f"    Bbox: {elem.get('bbox', 'No bbox')}")
        print(f"    X/Y: ({elem.get('x', 'No x')}, {elem.get('y', 'No y')})")
        print(f"    Width/Height: ({elem.get('width', 'No w')}, {elem.get('height', 'No h')})")
        print(f"    Center: ({elem.get('center_x', 'No cx')}, {elem.get('center_y', 'No cy')})")
        print()
    
    # Show elements with text
    text_elements = [e for e in parsed_elements if e.get("text", "").strip()]
    print(f"📝 Elements with text: {len(text_elements)}")
    
    for i, elem in enumerate(text_elements[:10]):  # Show first 10
        text = elem.get("text", "")[:40]  # Truncate long text
        bbox = elem.get("bbox", "N/A")
        x, y = elem.get("x", 0), elem.get("y", 0)
        center_x = elem.get("center_x", x)
        center_y = elem.get("center_y", y)
        print(f"  {i}: '{text}' bbox={bbox} at ({x}, {y}) center: ({center_x}, {center_y})")
    
    # Show elements by class
    class_elements = [e for e in parsed_elements if e.get("class", "").strip()]
    print(f"🏷️ Elements by class: {len(class_elements)}")
    
    class_counts = {}
    for elem in class_elements:
        class_name = elem.get("class", "")
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    for class_name, count in sorted(class_counts.items())[:5]:  # Show top 5 classes
        print(f"  {class_name}: {count} elements")
    
    # Look for TikTok-related elements specifically
    tiktok_elements = []
    for elem in parsed_elements:
        text = elem.get("text", "").lower()
        class_name = elem.get("class", "").lower()
        if "tiktok" in text or "tiktok" in class_name or "icon" in class_name or "app" in class_name:
            tiktok_elements.append(elem)
    
    if tiktok_elements:
        print(f"🎯 TikTok-related elements found: {len(tiktok_elements)}")
        for i, elem in enumerate(tiktok_elements):
            text = elem.get("text", "")
            bbox = elem.get("bbox", "N/A")
            print(f"  TikTok {i}: '{text}' bbox={bbox}")
            
            # Test coordinate calculation for TikTok elements
            if "bbox" in elem and len(elem["bbox"]) == 4:
                bbox = elem["bbox"]
                print(f"    TikTok {i} coordinate test:")
                if max(bbox) > 1.0:
                    print(f"      As absolute pixels: x={int(bbox[0])}, y={int(bbox[1])}")
                else:
                    if max(bbox) < 0.5:  # Likely center-based
                        center_x_rel = bbox[0]
                        center_y_rel = bbox[1]
                        half_width_rel = bbox[2] / 2
                        half_height_rel = bbox[3] / 2
                        corner_x = int((center_x_rel - half_width_rel) * 1080)
                        corner_y = int((center_y_rel - half_height_rel) * 2424)
                        center_x = int(center_x_rel * 1080)
                        center_y = int(center_y_rel * 2424)
                        print(f"      As center-based bbox: corner=({corner_x}, {corner_y}), center=({center_x}, {center_y})")
                    else:  # Likely corner-based
                        corner_x = int(bbox[0] * 1080)
                        corner_y = int(bbox[1] * 2424)
                        center_x = int((bbox[0] + bbox[2]) / 2 * 1080)
                        center_y = int((bbox[1] + bbox[3]) / 2 * 2424)
                        print(f"      As corner-based bbox: corner=({corner_x}, {corner_y}), center=({center_x}, {center_y})")
    
    # Test coordinate calculation for first few elements
    print(f"🧮 Testing coordinate calculation:")
    for i, elem in enumerate(parsed_elements[:3]):
        if "bbox" in elem and len(elem["bbox"]) == 4:
            bbox = elem["bbox"]
            print(f"  Element {i} bbox: {bbox}")
            
            # Test both relative and absolute interpretations
            if max(bbox) > 1.0:
                print(f"    As absolute pixels: x={int(bbox[0])}, y={int(bbox[1])}")
            else:
                print(f"    As relative (0-1): x={int(bbox[0] * 1080)}, y={int(bbox[1] * 2424)}")
                
                # Test center-based vs corner-based interpretation
                if max(bbox) < 0.5:  # Likely center-based
                    center_x_rel = bbox[0]
                    center_y_rel = bbox[1]
                    half_width_rel = bbox[2] / 2
                    half_height_rel = bbox[3] / 2
                    corner_x = int((center_x_rel - half_width_rel) * 1080)
                    corner_y = int((center_y_rel - half_height_rel) * 2424)
                    print(f"    As center-based bbox: corner=({corner_x}, {corner_y})")
                else:  # Likely corner-based
                    corner_x = int(bbox[0] * 1080)
                    corner_y = int(bbox[1] * 2424)
                    print(f"    As corner-based bbox: corner=({corner_x}, {corner_y})")
    
    print("--- End Debug ---\n")


def execute_task(
    state: DeploymentState, task: str, device: str, neo4j_db: Neo4jDatabase = None
) -> Dict[str, Any]:
    """
    Main function to execute a task

    Args:
        state: Initial state
        task: User task
        device: Device ID
        neo4j_db: Neo4j database connection (optional, uses global db by default)

    Returns:
        Execution result
    """
    # Update state using create_deployment_state function
    from data.State import create_deployment_state

    # Create new state
    state = create_deployment_state(task=task, device=device)

    # Use global db object
    neo4j_db = neo4j_db or db

    # Query database for all element nodes - using correct method name
    all_elements = neo4j_db.get_all_actions()
    if not all_elements:
        print("⚠️ No element nodes in database, falling back to React mode")
        state = fallback_to_react(state)
        return {"status": state["execution_status"], "state": state}

    # Query database for high-level actions related to the task
    high_level_actions = neo4j_db.get_high_level_actions_for_task(task)
    if high_level_actions:
        print(
            f"✓ Found {len(high_level_actions)} high-level actions related to the task"
        )

        # Check for shortcut associations
        shortcuts = check_shortcut_associations(state, high_level_actions)

        if shortcuts:
            print(f"✓ Found {len(shortcuts)} possible shortcuts")

            # Evaluate shortcut execution conditions
            valid_shortcuts = evaluate_shortcut_execution(state, shortcuts)

            if valid_shortcuts:
                print(f"✓ {len(valid_shortcuts)} shortcuts meet execution conditions")

                # Generate execution template
                execution_template = generate_execution_template(state, valid_shortcuts)

                if execution_template:
                    print("✓ Generated execution template")

                    # Sort shortcuts by priority
                    prioritized_shortcuts = prioritize_shortcuts(state, valid_shortcuts)

                    # Execute high-level operation
                    result = execute_high_level_action(
                        state, prioritized_shortcuts, execution_template
                    )

                    if result.get("status") == "success":
                        print(
                            f"✓ High-level operation executed successfully: {result.get('message', '')}"
                        )
                        state["execution_status"] = "success"
                        state["completed"] = True
                        return {"status": "success", "state": state}
                    else:
                        print(
                            f"❌ High-level operation execution failed: {result.get('message', '')}"
                        )
                        # Fall back to React mode on failure
                        state = fallback_to_react(state)
                        return {"status": state["execution_status"], "state": state}
        else:
            # No shortcuts, try executing basic operation sequence
            for action in high_level_actions:
                action_sequence = action.get("action_sequence", [])
                if not action_sequence:
                    continue

                # Capture and parse screen
                state = capture_and_parse_screen(state)
                if not state["current_page"]["screenshot"]:
                    state["retry_count"] += 1
                    if state["retry_count"] >= state["max_retries"]:
                        print(
                            f"❌ Failed to capture or parse screen {state['max_retries']} times in a row, falling back to React mode"
                        )
                        state = fallback_to_react(state)
                        return {"status": state["execution_status"], "state": state}
                    continue

                # Reset retry count
                state["retry_count"] = 0

                # Match screen elements
                element_matches = match_screen_elements(state, action_sequence)
                if not element_matches:
                    state["retry_count"] += 1
                    if state["retry_count"] >= state["max_retries"]:
                        print(
                            f"❌ No matching elements found {state['max_retries']} times in a row, falling back to React mode"
                        )
                        state = fallback_to_react(state)
                        return {"status": state["execution_status"], "state": state}
                    continue

                # Reset retry count
                state["retry_count"] = 0

                # Execute element action
                best_match = element_matches[0]
                success = execute_element_action(state, best_match)

                if success:
                    print(f"✓ Step {state['current_step']} executed successfully")

                    # Update history
                    state["history"].append(
                        {
                            "step": state["current_step"],
                            "screenshot": state["current_page"]["screenshot"],
                            "elements_json": state["current_page"]["elements_json"],
                            "action": best_match.get("action_type", "tap"),
                            "element_id": best_match.get("element_id", ""),
                            "screen_element_id": best_match.get(
                                "screen_element_id", -1
                            ),
                            "status": "success",
                        }
                    )

                    # Update current step
                    state["current_step"] += 1

                    # Check if all steps are completed
                    if state["current_step"] >= len(action_sequence):
                        print("✓ All steps completed")
                        state["execution_status"] = "success"
                        state["completed"] = True
                        return {"status": "success", "state": state}
                else:
                    print(f"❌ Step {state['current_step']} execution failed")

                    # Update history
                    state["history"].append(
                        {
                            "step": state["current_step"],
                            "screenshot": state["current_page"]["screenshot"],
                            "elements_json": state["current_page"]["elements_json"],
                            "action": best_match.get("action_type", "tap"),
                            "element_id": best_match.get("element_id", ""),
                            "screen_element_id": best_match.get(
                                "screen_element_id", -1
                            ),
                            "status": "error",
                        }
                    )

                    # Increment retry count
                    state["retry_count"] += 1
                    if state["retry_count"] >= state["max_retries"]:
                        print(
                            f"❌ Operation failed {state['max_retries']} times in a row, falling back to React mode"
                        )
                        state = fallback_to_react(state)
                        return {"status": state["execution_status"], "state": state}
    else:
        print("❌ No matching high-level actions found, falling back to React mode")
        state = fallback_to_react(state)
        return {"status": state["execution_status"], "state": state}

    # If all above methods fail, fall back to React mode
    print(
        "⚠️ Unable to complete task with high-level operations, falling back to basic operation space"
    )
    state = fallback_to_react(state)
    return {"status": state["execution_status"], "state": state}


def run_task_with_working_agent(task: str, device: str = "emulator-5554", use_clip: bool = True) -> Dict[str, Any]:
    """
    Execute a task using the WorkingUIAgent with CLIP capabilities
    
    Args:
        task: User task description
        device: Device ID
        use_clip: Whether to use CLIP-enhanced features
        
    Returns:
        Execution result
    """
    if not working_ui_agent:
        print("⚠️ WorkingUIAgent not available, falling back to standard execution")
        return run_task(task, device)
    
    try:
        print(f"🚀 Executing task with WorkingUIAgent: '{task}' on {device}")
        
        # Initialize agent for this device
        working_ui_agent.device_id = device
        
        # Take initial screenshot for analysis
        screenshot_path = working_ui_agent.take_screenshot("task_analysis")
        if not screenshot_path:
            print("❌ Failed to take initial screenshot")
            return {"status": "error", "message": "Screenshot failed"}
        
        # Analyze screenshot with CLIP if available
        if use_clip and working_ui_agent.clip_available:
            print("🧠 Analyzing screenshot with CLIP...")
            embedding = working_ui_agent.analyze_screenshot_with_clip(screenshot_path, task)
            if embedding is not None:
                print("✅ CLIP analysis completed")
        
        # Use CLIP to find elements matching the task description
        if "click" in task.lower() or "tap" in task.lower():
            element_coords = working_ui_agent.find_element_with_clip(task)
            if element_coords:
                x, y = element_coords
                print(f"🎯 Found target element at ({x}, {y}) using CLIP")
                
                # Wait for UI to be ready
                if working_ui_agent.wait_for_ui_ready():
                    # Perform the action
                    result = working_ui_agent.tap_element(x, y)
                    if result:
                        return {
                            "status": "success",
                            "message": f"Successfully executed: {task}",
                            "method": "working_ui_agent_clip",
                            "coordinates": [x, y],
                            "completed": True
                        }
        
        # If CLIP method didn't work, try coordinate estimation
        element_coords = working_ui_agent.find_element_with_clip(task)
        if element_coords:
            x, y = element_coords
            print(f"📍 Using coordinate estimation: ({x}, {y})")
            
            if working_ui_agent.wait_for_ui_ready():
                result = working_ui_agent.tap_element(x, y)
                if result:
                    return {
                        "status": "success", 
                        "message": f"Successfully executed: {task}",
                        "method": "working_ui_agent_coordinates",
                        "coordinates": [x, y],
                        "completed": True
                    }
        
        # Fallback to standard execution
        print("⚠️ WorkingUIAgent methods failed, falling back to standard execution")
        return run_task(task, device)
        
    except Exception as e:
        print(f"❌ WorkingUIAgent execution failed: {e}")
        return {
            "status": "error",
            "message": f"WorkingUIAgent error: {str(e)}",
            "fallback_available": True
        }


def run_task(task: str, device: str = "emulator-5554") -> Dict[str, Any]:
    """
    Execute a single task

    Args:
        task: User task description
        device: Device ID

    Returns:
        Execution result
    """
    print(f"🚀 Starting task execution: {task}")

    try:
        # Initialize state using create_deployment_state function
        from data.State import create_deployment_state

        state = create_deployment_state(
            task=task,
            device=device,
            max_retries=3,
        )

        # Execute task using LangGraph workflow
        workflow = build_workflow()
        app = workflow.compile()
        result = app.invoke(state)

        # Display final screenshot if execution was successful
        if (
            result["execution_status"] == "success"
            and result["current_page"]["screenshot"]
        ):
            try:
                from PIL import Image

                img = Image.open(result["current_page"]["screenshot"])
                img.show()
            except Exception as e:
                print(f"Unable to display final screenshot: {str(e)}")

        return {
            "status": result["execution_status"],
            "message": "Task execution completed",
            "steps_completed": result["current_step"],
            "total_steps": result["total_steps"],
            "completed": result.get("completed", False),  # Add completed status for frontend
        }

    except Exception as e:
        print(f"❌ Error executing task: {str(e)}")
        return {
            "status": "error",
            "message": f"Error executing task: {str(e)}",
            "error": str(e),
            "completed": False,  # Add completed status for frontend
        }


def check_shortcut_associations(
    state: DeploymentState, high_level_actions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Check if high-level actions are associated with shortcuts

    Args:
        state: Execution state
        high_level_actions: List of high-level actions

    Returns:
        List of associated shortcuts
    """
    print("🔍 Checking high-level action shortcut associations...")
    shortcuts = []

    for action in high_level_actions:
        action_id = action.get("action_id")
        if not action_id:
            continue

        # Query database for shortcuts associated with high-level action
        associated_shortcuts = state["neo4j_db"].get_shortcuts_for_action(action_id)
        if associated_shortcuts:
            for shortcut in associated_shortcuts:
                shortcuts.append(
                    {
                        "shortcut_id": shortcut.get("shortcut_id"),
                        "name": shortcut.get("name"),
                        "description": shortcut.get("description"),
                        "action_id": action_id,
                        "action_name": action.get("name"),
                        "action_sequence": action.get("action_sequence", []),
                        "conditions": shortcut.get("conditions", {}),
                        "priority": shortcut.get("priority", 0),
                        "page_flow": shortcut.get("page_flow", []),
                    }
                )

    if shortcuts:
        print(f"✓ Found {len(shortcuts)} associated shortcuts")
    else:
        print("⚠️ No associated shortcuts found")

    return shortcuts


def evaluate_shortcut_execution(
    state: DeploymentState, shortcuts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Evaluate if shortcuts meet execution conditions

    Args:
        state: Execution state
        shortcuts: List of shortcuts

    Returns:
        List of shortcuts that meet execution conditions
    """
    print("🧠 Evaluating shortcut execution conditions...")

    if not shortcuts:
        print("⚠️ No shortcuts to evaluate")
        return []

    # Prepare current screen information
    screen_desc = ""
    if state["current_page"]["elements_data"]:
        screen_desc = "Current screen contains the following elements:\n"
        for i, element in enumerate(state["current_page"]["elements_data"]):
            element_type = element.get("type", "Unknown type")
            element_content = element.get("content", "")
            screen_desc += f"{i+1}. Type: {element_type}, Content: {element_content}\n"

    # Prepare task information
    task_desc = state["task"]

    # Create evaluation prompt
    eval_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a smartphone operation assistant responsible for evaluating whether the current scenario meets the conditions for executing shortcuts.
Analyze the current screen information, user task, and shortcut execution conditions to determine which shortcuts can be executed.
Only execute shortcuts when their conditions match the current scenario.""",
            ),
            (
                "human",
                """User task: {task}

Current screen information:
{screen_info}

Available shortcuts:
{shortcuts_info}

Please evaluate if each shortcut meets execution conditions, return in JSON format:
{{
  "valid_shortcuts": [
    {{
      "shortcut_id": "ID of shortcut meeting conditions",
      "reason": "Reason for meeting conditions",
      "confidence": "Confidence level between 0.0-1.0"
    }},
    ...
  ]
}}

If no shortcuts meet conditions, return an empty list.
""",
            ),
        ]
    )

    # Prepare shortcuts information
    shortcuts_info = ""
    for i, shortcut in enumerate(shortcuts):
        shortcuts_info += f"{i+1}. ID: {shortcut.get('shortcut_id')}\n"
        shortcuts_info += f"   Name: {shortcut.get('name')}\n"
        shortcuts_info += (
            f"   Description: {shortcut.get('description', 'No description')}\n"
        )

        # Add conditions information
        conditions = shortcut.get("conditions", {})
        if conditions:
            shortcuts_info += "   Execution conditions:\n"
            if isinstance(conditions, dict):
                for k, v in conditions.items():
                    shortcuts_info += f"     - {k}: {v}\n"
            elif isinstance(conditions, str):
                shortcuts_info += f"     - {conditions}\n"

        shortcuts_info += "\n"

    # Call LLM for evaluation
    try:
        # Prepare input
        eval_input = {
            "task": task_desc,
            "screen_info": screen_desc,
            "shortcuts_info": shortcuts_info,
        }

        # Create parser
        parser = JsonOutputParser()

        # Build chain
        eval_chain = eval_prompt | model | parser

        # Execute evaluation
        result = eval_chain.invoke(eval_input)

        # Parse results
        valid_shortcuts = result.get("valid_shortcuts", [])

        if valid_shortcuts:
            # Find corresponding complete shortcut information
            valid_shortcut_objects = []
            for valid in valid_shortcuts:
                shortcut_id = valid.get("shortcut_id")
                for shortcut in shortcuts:
                    if shortcut.get("shortcut_id") == shortcut_id:
                        # Add evaluation information
                        shortcut_copy = shortcut.copy()
                        shortcut_copy["evaluation"] = {
                            "reason": valid.get("reason", ""),
                            "confidence": valid.get("confidence", 0.0),
                        }
                        valid_shortcut_objects.append(shortcut_copy)
                        break

            print(
                f"✓ Found {len(valid_shortcut_objects)} shortcuts meeting execution conditions"
            )
            return valid_shortcut_objects
        else:
            print("⚠️ No shortcuts meet execution conditions")
            return []

    except Exception as e:
        print(f"❌ Error evaluating shortcut execution conditions: {str(e)}")
        return []


def generate_execution_template(
    state: DeploymentState, shortcuts: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate execution template based on shortcuts

    Args:
        state: Execution state
        shortcuts: List of shortcuts that meet execution conditions

    Returns:
        Execution template with operation steps and parameters
    """
    print("📝 Generating execution template...")

    if not shortcuts:
        print("⚠️ No available shortcuts, cannot generate execution template")
        return {}

    # Select shortcut with highest confidence
    selected_shortcut = max(
        shortcuts, key=lambda x: x.get("evaluation", {}).get("confidence", 0)
    )

    # Get device dimensions
    device_size = get_device_size.invoke(state["device"])
    if isinstance(device_size, str):
        device_size = {"width": 1080, "height": 1920}

    # Prepare current screen information
    screen_elements = state["current_page"]["elements_data"]
    screen_elements_json = json.dumps(screen_elements, ensure_ascii=False, indent=2)

    # Create template generation prompt
    template_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a smartphone operation assistant responsible for generating detailed execution templates based on shortcuts and current screen state.
The execution template should include all steps needed to complete the operation, with each step specifying the action type, target element, and necessary parameters.
Available action types include: tap, text (input text), swipe, long_press, back.
Ensure the generated template can accurately execute the operations described in the shortcut.""",
            ),
            (
                "human",
                """Shortcut information:
{shortcut_info}

Current screen elements:
{screen_elements}

Device size: {device_size}

Please generate a detailed template for executing this shortcut, return in JSON format:
{
  "steps": [
    {
      "action_type": "action type(tap/text/swipe/long_press/back)",
      "target_element_id": target element ID(number),
      "parameters": {
        // Add appropriate parameters based on action type
        // e.g., text operation needs "text" parameter
        // swipe operation needs "direction" and "distance" parameters
      }
    },
    // more steps...
  ]
}

Ensure each step has a clear operation target and necessary parameters. If the operation doesn't need a target element (like back operation), you can omit target_element_id.
""",
            ),
        ]
    )

    # Prepare shortcut information
    shortcut_info = f"ID: {selected_shortcut.get('shortcut_id')}\n"
    shortcut_info += f"Name: {selected_shortcut.get('name')}\n"
    shortcut_info += (
        f"Description: {selected_shortcut.get('description', 'No description')}\n"
    )

    # Add action sequence information
    action_sequence = selected_shortcut.get("action_sequence", [])
    if action_sequence:
        shortcut_info += "Action sequence:\n"
        if isinstance(action_sequence, list):
            for i, action in enumerate(action_sequence):
                shortcut_info += f"  {i+1}. {json.dumps(action, ensure_ascii=False)}\n"
        elif isinstance(action_sequence, str):
            shortcut_info += f"  {action_sequence}\n"

    # Add evaluation information
    evaluation = selected_shortcut.get("evaluation", {})
    if evaluation:
        shortcut_info += f"Execution reason: {evaluation.get('reason', 'None')}\n"
        shortcut_info += f"Confidence: {evaluation.get('confidence', 0.0)}\n"

    # Call LLM to generate template
    try:
        # Prepare input
        template_input = {
            "shortcut_info": shortcut_info,
            "screen_elements": screen_elements_json,
            "device_size": json.dumps(device_size, ensure_ascii=False),
        }

        # Create parser
        parser = JsonOutputParser()

        # Build chain
        template_chain = template_prompt | model | parser

        # Execute generation
        result = template_chain.invoke(template_input)

        # Validate result
        if (
            "steps" in result
            and isinstance(result["steps"], list)
            and len(result["steps"]) > 0
        ):
            print(
                f"✓ Successfully generated execution template with {len(result['steps'])} steps"
            )
            return result
        else:
            print("❌ Generated execution template is invalid")
            return {}

    except Exception as e:
        print(f"❌ Error generating execution template: {str(e)}")
        return {}


def prioritize_shortcuts(
    state: Dict[str, Any], shortcuts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Prioritize shortcuts based on page flow

    Args:
        state: Execution state
        shortcuts: List of shortcuts

    Returns:
        Prioritized list of shortcuts
    """
    if not shortcuts or len(shortcuts) <= 1:
        return shortcuts

    try:
        # Get page flow information from database
        page_flow = state["neo4j_db"].get_page_flow()

        if not page_flow:
            print("⚠️ No page flow information found, using default sorting")
            # Default sort by match score
            return sorted(
                shortcuts,
                key=lambda x: x["element_match"].get("match_score", 0),
                reverse=True,
            )

        # Assign priority to each shortcut based on page flow position
        prioritized = []
        for shortcut in shortcuts:
            # Find shortcut position in page flow
            position = -1
            shortcut_id = shortcut["shortcut_id"]

            for idx, flow_node in enumerate(page_flow):
                if flow_node.get("shortcut_id") == shortcut_id:
                    position = idx
                    break

            prioritized.append(
                {
                    "shortcut": shortcut,
                    "flow_position": position,
                    "match_score": shortcut["element_match"].get("match_score", 0),
                }
            )

        # First sort by flow position, unknown positions (-1) at the end
        # For same positions, sort by match score
        prioritized.sort(
            key=lambda x: (
                x["flow_position"] if x["flow_position"] >= 0 else float("inf"),
                -x["match_score"],
            )
        )

        return [item["shortcut"] for item in prioritized]

    except Exception as e:
        print(f"⚠️ Shortcut prioritization failed: {str(e)}")
        # Return original list on error
        return shortcuts


def execute_high_level_action(
    state: DeploymentState,
    shortcuts: List[Dict[str, Any]],
    execution_template: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute high-level operations

    Args:
        state: Execution state
        shortcuts: List of shortcuts meeting execution conditions
        execution_template: Execution template

    Returns:
        Execution result
    """
    print("🚀 Executing high-level operations...")

    if not execution_template or "steps" not in execution_template:
        print("❌ Invalid execution template")
        return {"status": "error", "message": "Invalid execution template"}

    steps = execution_template["steps"]
    if not steps or not isinstance(steps, list):
        print("❌ No valid steps in execution template")
        return {"status": "error", "message": "No valid steps in execution template"}

    # Initialize execution state
    state["current_step"] = 0
    state["total_steps"] = len(steps)
    state["execution_status"] = "running"
    state["history"] = []

    # Record shortcut execution start
    shortcut_names = ", ".join([s.get("name", "Unnamed shortcut") for s in shortcuts])
    print(f"Starting execution of shortcuts: {shortcut_names}")
    print(f"Total steps: {state['total_steps']}")

    # Execute each step
    while state["current_step"] < state["total_steps"]:
        current_step_idx = state["current_step"]
        step = steps[current_step_idx]

        print(f"\nExecuting step {current_step_idx + 1}/{state['total_steps']}")

        # Capture and parse current screen
        state = capture_and_parse_screen(state)
        if not state["current_page"]["screenshot"]:
            state["retry_count"] += 1
            if state["retry_count"] >= state["max_retries"]:
                print(
                    f"❌ Failed to capture or parse screen {state['max_retries']} times in a row"
                )
                return {
                    "status": "error",
                    "message": "Unable to capture or parse screen",
                }

            # Wait a second before retrying
            time.sleep(1)
            continue

        # Reset retry counter
        state["retry_count"] = 0

        # Get action type and parameters
        action_type = step.get("action_type", "tap")
        target_element_id = step.get("target_element_id")
        parameters = step.get("parameters", {})

        # Special handling for back operation
        if action_type == "back":
            print("Executing back operation")
            result = screen_action.invoke({"device": state["device"], "action": "back"})

            # Record history
            state["history"].append(
                {
                    "step": current_step_idx,
                    "screenshot": state["current_page"]["screenshot"],
                    "elements_json": state["current_page"]["elements_json"],
                    "action": "back",
                    "status": "success",
                }
            )

            # Move to next step
            state["current_step"] += 1
            time.sleep(1)  # Wait for operation to take effect
            continue

        # For operations requiring target element
        if target_element_id is None:
            print("❌ Operation missing target element ID")
            return {
                "status": "error",
                "message": f"Step {current_step_idx + 1} missing target element ID",
            }

        # Check if target element exists
        screen_elements = state["current_page"]["elements_data"]
        if target_element_id < 0 or target_element_id >= len(screen_elements):
            print(f"❌ Invalid target element ID: {target_element_id}")
            return {
                "status": "error",
                "message": f"Invalid target element ID for step {current_step_idx + 1}",
            }

        # Get element position
        element = screen_elements[target_element_id]
        bbox = element.get("bbox", [0, 0, 0, 0])

        # Get device size and calculate center point with better precision
        device_size = get_device_size.invoke(state["device"])
        if isinstance(device_size, str):
            device_size = {"width": 1080, "height": 1920}

        # Get device dimensions dynamically
        device_size = get_device_size.invoke(state["device"])
        if isinstance(device_size, str):
            if "x" in device_size:
                try:
                    width, height = map(int, device_size.split("x"))
                except:
                    width, height = 1080, 2424  # Default fallback
            else:
                width, height = 1080, 2424
        else:
            width = device_size.get("width", 1080)
            height = device_size.get("height", 2424)
        
        # Calculate center point with bounds checking
        center_x = max(10, min(width - 10, int((bbox[0] + bbox[2]) / 2 * width)))
        center_y = max(10, min(height - 10, int((bbox[1] + bbox[3]) / 2 * height)))
        
        print(f"🎯 Calculated action coordinates: ({center_x}, {center_y}) for element with bbox {bbox}")
        print(f"📱 Device dimensions: {width}x{height}, coordinates are absolute pixels")

        # Prepare operation parameters
        action_params = {
            "device": state["device"],
            "action": action_type,
            "x": center_x,
            "y": center_y,
        }

        # Add specific parameters based on action type
        if action_type == "text":
            action_params["input_str"] = parameters.get("text", "")
        elif action_type == "long_press":
            action_params["duration"] = parameters.get("duration", 1000)
        elif action_type == "swipe":
            action_params["direction"] = parameters.get("direction", "up")
            action_params["dist"] = parameters.get("distance", "medium")

        # Execute operation
        print(
            f"Executing operation: {action_type} at position ({center_x}, {center_y})"
        )
        if action_type == "text":
            print(f"Input text: {action_params.get('input_str', '')}")

        result = screen_action.invoke(action_params)

        # Parse operation result
        success = False
        if isinstance(result, str):
            try:
                result_json = json.loads(result)
                if result_json.get("status") == "success":
                    success = True
            except:
                pass

        if success:
            print(f"✓ Step {current_step_idx + 1} executed successfully")

            # Record history
            state["history"].append(
                {
                    "step": current_step_idx,
                    "screenshot": state["current_page"]["screenshot"],
                    "elements_json": state["current_page"]["elements_json"],
                    "action": action_type,
                    "target_element_id": target_element_id,
                    "parameters": parameters,
                    "status": "success",
                }
            )

            # Move to next step
            state["current_step"] += 1
            state["retry_count"] = 0

            # Wait for operation to take effect
            time.sleep(1.5)
        else:
            print(f"❌ Step {current_step_idx + 1} execution failed")

            # Record history
            state["history"].append(
                {
                    "step": current_step_idx,
                    "screenshot": state["current_page"]["screenshot"],
                    "elements_json": state["current_page"]["elements_json"],
                    "action": action_type,
                    "target_element_id": target_element_id,
                    "parameters": parameters,
                    "status": "error",
                }
            )

            # Increment retry counter
            state["retry_count"] += 1
            if state["retry_count"] >= state["max_retries"]:
                print(f"❌ Operation failed {state['max_retries']} times in a row")
                return {
                    "status": "error",
                    "message": f"Step {current_step_idx + 1} execution failed",
                }

            # Wait a second before retrying
            time.sleep(1)

    # Capture final screen
    state = capture_and_parse_screen(state)

    # Execution complete
    print(
        f"\n✨ High-level operation execution complete! Completed {state['current_step']} operations"
    )
    state["execution_status"] = "success"
    state["completed"] = True

    return {
        "status": "success",
        "message": "Successfully executed high-level operations",
        "steps_completed": state["current_step"],
        "total_steps": state["total_steps"],
        "final_screenshot": state["current_page"]["screenshot"],
        "execution_history": state["history"],
    }


def capture_screen_node(state: DeploymentState) -> DeploymentState:
    print("📸 Capturing and parsing current screen...")

    state_dict = dict(state)
    updated_state = capture_and_parse_screen(state_dict)

    # Update state
    for key, value in updated_state.items():
        if key in state:
            state[key] = value

    if not state["current_page"]["screenshot"]:
        state["should_fallback"] = True
        print("❌ Unable to capture screen, marking for fallback")
    else:
        print("✓ Screen captured successfully")

    return state


def match_elements_node(state: DeploymentState) -> DeploymentState:
    """
    Match current screen elements using visual embeddings
    """
    print("🔍 Matching current screen elements using visual embeddings...")

    # Get all element nodes from database - using correct method name
    all_elements = db.get_all_actions()
    if not all_elements:
        print("⚠️ No element nodes in database, marking for fallback")
        state["should_fallback"] = True
        return state

    # Build action sequence with all elements from database
    action_sequence = []
    for element in all_elements:
        # Ensure element has element_id field
        if "element_id" in element:
            action_sequence.append(
                {
                    "element_id": element["element_id"],
                    "atomic_action": "tap",  # Default action
                    "action_params": {},
                }
            )
        else:
            # If no element_id, try using other possible ID fields
            element_id = (
                element.get("id") or element.get("node_id") or str(hash(str(element)))
            )
            print(
                f"⚠️ Element missing element_id field, using alternative ID: {element_id}"
            )
            action_sequence.append(
                {
                    "element_id": element_id,
                    "atomic_action": "tap",  # Default action
                    "action_params": {},
                }
            )

    # Call match_screen_elements function
    state_dict = dict(state)
    element_matches = match_screen_elements(state_dict, action_sequence)
    state["matched_elements"] = element_matches

    if not state["matched_elements"]:
        print(
            "⚠️ No matching screen elements found, trying high-level task matching first"
        )

        # Try matching task to high-level actions
        is_matched, matched_action = match_task_to_action(state_dict, state["task"])

        if is_matched and matched_action:
            print(
                f"✓ Task matched to high-level action: {matched_action.get('name', 'Unknown')}"
            )
            # Save current executing high-level action
            state["current_action"] = matched_action

            # Get element sequence
            element_sequence = matched_action.get("element_sequence", [])
            if isinstance(element_sequence, str):
                try:
                    element_sequence = json.loads(element_sequence)
                except:
                    print(f"❌ Failed to parse element sequence")
                    state["should_fallback"] = True
                    return state

            if not element_sequence or not isinstance(element_sequence, list):
                print(f"❌ Element sequence is empty or invalid format")
                state["should_fallback"] = True
                return state

            # Update step information in state
            state["current_step"] = 0
            state["total_steps"] = len(element_sequence)

            # Recapture screen and match elements
            updated_state = capture_and_parse_screen(state_dict)
            for key, value in updated_state.items():
                if key in state:
                    state[key] = value

            element_matches = match_screen_elements(state_dict, element_sequence)
            state["matched_elements"] = element_matches

            if not element_matches:
                print(
                    "❌ Still no matching screen elements found, marking for fallback"
                )
                state["should_fallback"] = True
        else:
            print("❌ No matching high-level actions found, marking for fallback")
            state["should_fallback"] = True
    else:
        print(f"✓ Found {len(state['matched_elements'])} matching elements")

    return state


def check_shortcuts_node(state: DeploymentState) -> DeploymentState:
    """
    Check element associations with shortcuts
    """
    print("🔍 Checking element associations with shortcuts...")

    if not state["matched_elements"]:
        print("⚠️ No matched elements, cannot check shortcut associations")
        state["should_fallback"] = True
        return state

    # Call check_shortcut_associations function
    state_dict = dict(state)
    associated_shortcuts = check_shortcut_associations(
        state_dict, state["matched_elements"]
    )
    state["associated_shortcuts"] = associated_shortcuts

    if state["associated_shortcuts"]:
        print(f"✓ Found {len(state['associated_shortcuts'])} associated shortcut nodes")

        # Priority sorting
        prioritized_shortcuts = prioritize_shortcuts(state_dict, associated_shortcuts)
        state["associated_shortcuts"] = prioritized_shortcuts
    else:
        print("📝 No associated shortcut nodes found")

    return state


def shortcut_evaluation_node(state: DeploymentState) -> DeploymentState:
    """
    Evaluate whether to execute shortcut operations
    """
    print("🧠 Evaluating whether to execute shortcut operations...")

    if not state["associated_shortcuts"]:
        print("⚠️ No associated shortcuts, skipping evaluation")
        return state

    # Call evaluate_shortcut_execution function
    state_dict = dict(state)
    execution_decision = evaluate_shortcut_execution(
        state_dict, state["associated_shortcuts"], state["task"]
    )

    state["should_execute_shortcut"] = execution_decision.get("should_execute", False)
    if "shortcut" in execution_decision:
        state["current_shortcut"] = execution_decision["shortcut"]

    if state["should_execute_shortcut"]:
        print(
            f"✓ Decided to execute shortcut: {state['current_shortcut'].get('name', 'Unknown')}"
        )
        print(f"  Reason: {execution_decision.get('reason', '')}")
    else:
        print(
            f"⚠️ Decided not to execute shortcut operations: {execution_decision.get('reason', '')}"
        )

    return state


def generate_template_node(state: DeploymentState) -> DeploymentState:
    """
    Generate execution template
    """
    print("📝 Generating execution template...")

    if not state["should_execute_shortcut"] or "current_shortcut" not in state:
        print("⚠️ Not executing shortcut, skipping template generation")
        return state

    # Call generate_execution_template function
    state_dict = dict(state)
    execution_template = generate_execution_template(
        state_dict, state["current_shortcut"], state["task"]
    )
    state["execution_template"] = execution_template

    print(
        f"✓ Generated execution template with {len(state['execution_template']['steps'])} steps"
    )

    return state


def execute_action_node(state: DeploymentState) -> DeploymentState:
    """
    Execute operation
    """
    state_dict = dict(state)

    if state["should_execute_shortcut"] and state["execution_template"]:
        print("🚀 Executing high-level operation...")
        # Call execute_high_level_action function
        result = execute_high_level_action(
            state_dict, state["associated_shortcuts"], state["execution_template"]
        )

        if result["status"] == "success":
            print("✨ High-level operation executed successfully!")
            state["execution_status"] = "success"
            state["completed"] = True

            # Update history
            if "execution_history" in result:
                state["history"] = result["execution_history"]

            # Update final screenshot
            if "final_screenshot" in result and "current_page" in state:
                state["current_page"]["screenshot"] = result["final_screenshot"]
        else:
            print(
                f"❌ High-level operation execution failed: {result.get('message', '')}"
            )
            # Mark for fallback on failure
            state["should_fallback"] = True
    else:
        print("📝 Attempting to match task with high-level actions...")
        # Call match_task_to_action function
        is_matched, matched_action = match_task_to_action(state_dict, state["task"])

        if is_matched and matched_action:
            print(
                f"✓ Task matched to high-level action: {matched_action.get('name', 'Unknown')}"
            )

            # Save current executing high-level action
            state["current_action"] = matched_action

            # Get action sequence
            element_sequence = matched_action.get("element_sequence", [])
            if isinstance(element_sequence, str):
                try:
                    element_sequence = json.loads(element_sequence)
                except:
                    print(f"❌ Failed to parse element sequence")
                    state["should_fallback"] = True
                    return state

            if not element_sequence or not isinstance(element_sequence, list):
                print(f"❌ Element sequence is empty or incorrectly formatted")
                state["should_fallback"] = True
                return state

            # Update state
            state["current_step"] = 0
            state["total_steps"] = len(element_sequence)
            state["execution_status"] = "running"

            # Execute operation sequence (should actually enter next cycle)
            state["execution_template"] = {"steps": element_sequence}
            # Not executing here, returning to capture_screen for next cycle
        else:
            print(
                "⚠️ Unable to complete task with high-level operations, marking for fallback"
            )
            state["should_fallback"] = True

    return state


def fallback_node(state: DeploymentState) -> DeploymentState:
    """
    Fall back to React mode
    """
    print("⚠️ Falling back to basic operation space")

    # Call fallback_to_react function
    state = fallback_to_react(state)

    # Mark task as completed
    state["completed"] = True

    return state


# Routing functions
def should_fallback(state: DeploymentState) -> str:
    """
    Decide whether to fall back to basic operations
    """
    if state["should_fallback"]:
        return "fallback"
    return "continue"


def should_execute_shortcut(state: DeploymentState) -> str:
    """
    Decide whether to execute shortcut
    """
    if state["should_execute_shortcut"]:
        return "execute_shortcut"
    return "match_task"


def is_task_completed(state: DeploymentState) -> str:
    """
    Check if task is completed
    """
    if state["completed"]:
        return "end"
    return "continue"


# Build state graph
def build_workflow() -> StateGraph:
    """
    Build workflow state graph
    """
    workflow = StateGraph(DeploymentState)

    # Add nodes
    workflow.add_node("capture_screen", capture_screen_node)
    workflow.add_node("match_elements", match_elements_node)
    workflow.add_node("check_shortcuts", check_shortcuts_node)
    workflow.add_node("evaluate_shortcut", shortcut_evaluation_node)
    workflow.add_node("generate_template", generate_template_node)
    workflow.add_node("execute_action", execute_action_node)
    workflow.add_node("fallback", fallback_node)
    workflow.add_node(
        "check_completion", check_task_completion
    )  # New task completion check node

    # Define edges
    workflow.set_entry_point("capture_screen")

    # Routing after screen capture
    workflow.add_conditional_edges(
        "capture_screen",
        should_fallback,
        {"fallback": "fallback", "continue": "match_elements"},
    )

    # Routing after element matching
    workflow.add_conditional_edges(
        "match_elements",
        should_fallback,
        {"fallback": "fallback", "continue": "check_shortcuts"},
    )

    # Check shortcut associations
    workflow.add_edge("check_shortcuts", "evaluate_shortcut")

    # Routing after shortcut evaluation
    workflow.add_conditional_edges(
        "evaluate_shortcut",
        should_execute_shortcut,
        {"execute_shortcut": "generate_template", "match_task": "execute_action"},
    )

    # Execute action after template generation
    workflow.add_edge("generate_template", "execute_action")

    # Check task completion after action execution
    workflow.add_edge("execute_action", "check_completion")

    # Routing after task completion check
    workflow.add_conditional_edges(
        "check_completion",
        is_task_completed,
        {"end": END, "continue": "capture_screen"},
    )

    # Check task completion after fallback
    workflow.add_edge("fallback", "check_completion")

    return workflow


def check_task_completion(state: DeploymentState) -> DeploymentState:
    """
    Determine if task is completed

    Args:
        state: Execution state

    Returns:
        Updated execution state with task completion status
    """
    # Skip judgment if too few steps, but check for obvious completion patterns
    if state["current_step"] < 1:
        return state
    
    # Quick check: if we've done the same action multiple times recently, likely completed or stuck
    recent_actions = []
    if len(state["history"]) >= 2:  # Check after just 2 actions instead of 3
        recent_actions = [step.get("action", "") for step in state["history"][-2:]]
        if len(set(recent_actions)) == 1 and recent_actions[0] in ["swipe", "tap", "react_mode"]:
            # Same action repeated 2 times - likely stuck or task completed
            print(f"🔄 Detected repeated action '{recent_actions[0]}' - task likely completed")
            state["completed"] = True
            return state
    
    # Special check for simple single-action tasks
    if len(state["history"]) >= 1:
        last_action = state["history"][-1]
        task_lower = state["task"].lower()
        
        # For simple "click" or "tap" tasks, one successful action should be enough
        if any(word in task_lower for word in ["click", "tap", "press", "select"]):
            if last_action.get("status") == "success" and last_action.get("action") in ["tap", "react_mode"]:
                print(f"🎯 Simple click task completed successfully")
                state["completed"] = True
                return state

    print("🔍 Evaluating if task is completed...")

    # Get task description
    task = state["task"]

    # Step 1: Generate task completion criteria
    completion_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant that will help analyze task completion criteria. Please carefully read the following user task:",
            ),
            (
                "human",
                f"The user's task is: {task}\nPlease describe clear, checkable task completion criteria. For example: 'When certain elements or states appear on the page, it indicates the task is complete.'",
            ),
        ]
    )

    completion_chain = completion_prompt | model | StrOutputParser()
    completion_criteria = completion_chain.invoke({})

    # Collect recent screenshots
    recent_screenshots = []
    for step in state["history"][-3:]:
        if "screenshot" in step and step["screenshot"]:
            recent_screenshots.append(step["screenshot"])

    if not recent_screenshots:
        if state["current_page"]["screenshot"]:
            recent_screenshots.append(state["current_page"]["screenshot"])

    if not recent_screenshots:
        print("⚠️ No screenshots available, cannot determine if task is complete")
        return state

    # Build image messages
    image_messages = []
    for idx, img_path in enumerate(recent_screenshots, start=1):
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            image_messages.append(
                HumanMessage(
                    content=[
                        {"type": "text", "text": f"Here is data for screenshot {idx}:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_data}"},
                        },
                    ]
                )
            )

    # Step 2: Determine if task is complete
    judgement_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a page assessment assistant that will determine if a task is complete based on completion criteria and current page screenshots. Please only respond with 'yes' or 'no'.",
            ),
            (
                "human",
                f"The completion criteria is: {completion_criteria}\n"
                f"Based on the following screenshots, determine if the task is complete. Note that if screenshots are identical, it may indicate the task cannot proceed, so respond with 'yes' to end the program.",
            ),
        ]
    )

    # Create the judgement chain and invoke it
    judgement_chain = judgement_prompt | model | StrOutputParser()
    
    # Prepare input for the chain
    judgement_input = {
        "completion_criteria": completion_criteria,
    }
    
    # Call LLM for judgment
    judgement_answer = judgement_chain.invoke(judgement_input).strip()

    # Update task completion status
    if "yes" in judgement_answer.lower() or "complete" in judgement_answer.lower():
        state["completed"] = True
        state["execution_status"] = "completed"
        print(f"✓ Task completed: {judgement_answer}")
    else:
        state["completed"] = False
        print(f"⚠️ Task not completed: {judgement_answer}")

    # Add to history
    state["history"].append(
        {
            "step": state["current_step"],
            "action": "task_completion_check",
            "completion_criteria": completion_criteria,
            "judgement": judgement_answer,
            "status": "success",
            "completed": state["completed"],
        }
    )

    return state
