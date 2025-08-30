#!/usr/bin/env python3
"""
Fix for Gemini schema conversion issue with tuple parameters.
This patches the LangChain Google GenAI function utils to properly handle prefixItems.
"""

import google.ai.generativelanguage as glm
from typing import Any, Dict, Union, List

def _get_properties_from_schema_fixed(schema: Dict) -> Dict[str, Any]:
    """Fixed version that handles prefixItems for tuples."""
    properties: Dict[str, Dict[str, Union[str, int, Dict, List]]] = {}
    for k, v in schema.items():
        if not isinstance(k, str):
            print(f"Key '{k}' is not supported in schema, type={type(k)}")
            continue
        if not isinstance(v, Dict):
            print(f"Value '{v}' is not supported in schema, ignoring v={v}")
            continue
        properties_item: Dict[str, Union[str, int, Dict, List]] = {}
        
        # Handle anyOf (e.g., Optional types)
        if v.get("anyOf") and all(
            anyOf_type.get("type") != "null" for anyOf_type in v.get("anyOf", [])
        ):
            properties_item["anyOf"] = [
                _format_json_schema_to_gapic_fixed(anyOf_type)
                for anyOf_type in v.get("anyOf", [])
            ]
        elif v.get("type") or v.get("anyOf") or v.get("type_"):
            item_type_ = _get_type_from_schema_fixed(v)
            properties_item["type_"] = item_type_
            if _is_nullable_schema_fixed(v):
                properties_item["nullable"] = True

            # Replace `v` with chosen definition for array / object json types
            any_of_types = v.get("anyOf")
            if any_of_types and item_type_ in [glm.Type.ARRAY, glm.Type.OBJECT]:
                json_type_ = "array" if item_type_ == glm.Type.ARRAY else "object"
                # Use Index -1 for consistency with `_get_nullable_type_from_schema`
                v = [val for val in any_of_types if val.get("type") == json_type_][-1]

        if v.get("enum"):
            properties_item["enum"] = v["enum"]

        description = v.get("description")
        if description and isinstance(description, str):
            properties_item["description"] = description

        # FIXED: Handle both 'items' and 'prefixItems' for arrays
        if properties_item.get("type_") == glm.Type.ARRAY:
            if v.get("items"):
                properties_item["items"] = _get_items_from_schema_any_fixed(v.get("items"))
            elif v.get("prefixItems"):
                # Handle prefixItems (used for tuples)
                # For tuples, we'll use the first item type as the array item type
                prefix_items = v.get("prefixItems", [])
                if prefix_items:
                    # For Tuple[int, int], use integer as the item type
                    first_item = prefix_items[0]
                    properties_item["items"] = _get_items_from_schema_any_fixed(first_item)

        if properties_item.get("type_") == glm.Type.OBJECT:
            if (
                v.get("anyOf")
                and isinstance(v["anyOf"], list)
                and isinstance(v["anyOf"][0], dict)
            ):
                v = v["anyOf"][0]
            v_properties = v.get("properties")
            if v_properties:
                properties_item["properties"] = _get_properties_from_schema_any_fixed(
                    v_properties
                )
                if isinstance(v_properties, dict):
                    properties_item["required"] = [
                        k for k, v in v_properties.items() if "default" not in v
                    ]
            else:
                # Providing dummy type for object without properties
                properties_item["type_"] = glm.Type.STRING

        if k == "title" and "description" not in properties_item:
            properties_item["description"] = k + " is " + str(v)

        properties[k] = properties_item

    return properties

def _get_properties_from_schema_any_fixed(schema: Any) -> Dict[str, Any]:
    if isinstance(schema, Dict):
        return _get_properties_from_schema_fixed(schema)
    return {}

def _get_items_from_schema_any_fixed(schema: Any) -> Dict[str, Any]:
    if isinstance(schema, (dict, list, str)):
        return _get_items_from_schema_fixed(schema)
    return {}

def _get_items_from_schema_fixed(schema: Union[Dict, List, str]) -> Dict[str, Any]:
    items: Dict = {}
    if isinstance(schema, List):
        for i, v in enumerate(schema):
            items[f"item{i}"] = _get_properties_from_schema_any_fixed(v)
    elif isinstance(schema, Dict):
        items["type_"] = _get_type_from_schema_fixed(schema)
        if items["type_"] == glm.Type.OBJECT and "properties" in schema:
            items["properties"] = _get_properties_from_schema_any_fixed(schema["properties"])
        if items["type_"] == glm.Type.ARRAY and "items" in schema:
            items["items"] = _format_json_schema_to_gapic_fixed(schema["items"])
        if "title" in schema or "description" in schema:
            items["description"] = (
                schema.get("description") or schema.get("title") or ""
            )
        if _is_nullable_schema_fixed(schema):
            items["nullable"] = True
        if "required" in schema:
            items["required"] = schema["required"]
    else:
        # str
        items["type_"] = _get_type_from_schema_fixed({"type": schema})
        if _is_nullable_schema_fixed({"type": schema}):
            items["nullable"] = True

    return items

def _get_type_from_schema_fixed(schema: Dict[str, Any]) -> int:
    return _get_nullable_type_from_schema_fixed(schema) or glm.Type.STRING

def _get_nullable_type_from_schema_fixed(schema: Dict[str, Any]) -> int:
    TYPE_ENUM = {
        "string": glm.Type.STRING,
        "number": glm.Type.NUMBER,
        "integer": glm.Type.INTEGER,
        "boolean": glm.Type.BOOLEAN,
        "array": glm.Type.ARRAY,
        "object": glm.Type.OBJECT,
        "null": None,
    }
    
    if "anyOf" in schema:
        types = [
            _get_nullable_type_from_schema_fixed(sub_schema) for sub_schema in schema["anyOf"]
        ]
        types = [t for t in types if t is not None]  # Remove None values
        if types:
            return types[-1]  # TODO: update FunctionDeclaration and pass all types?
        else:
            pass
    elif "type" in schema or "type_" in schema:
        type_ = schema["type"] if "type" in schema else schema["type_"]
        if isinstance(type_, int):
            return type_
        stype = str(schema["type"]) if "type" in schema else str(schema["type_"])
        return TYPE_ENUM.get(stype, glm.Type.STRING)
    else:
        pass
    return glm.Type.STRING  # Default to string if no valid types found

def _is_nullable_schema_fixed(schema: Dict[str, Any]) -> bool:
    TYPE_ENUM = {
        "string": glm.Type.STRING,
        "number": glm.Type.NUMBER,
        "integer": glm.Type.INTEGER,
        "boolean": glm.Type.BOOLEAN,
        "array": glm.Type.ARRAY,
        "object": glm.Type.OBJECT,
        "null": None,
    }
    
    if "anyOf" in schema:
        types = [
            _get_nullable_type_from_schema_fixed(sub_schema) for sub_schema in schema["anyOf"]
        ]
        return any(t is None for t in types)
    elif "type" in schema or "type_" in schema:
        type_ = schema["type"] if "type" in schema else schema["type_"]
        if isinstance(type_, int):
            return False
        stype = str(schema["type"]) if "type" in schema else str(schema["type_"])
        return TYPE_ENUM.get(stype, glm.Type.STRING) is None
    else:
        pass
    return False

def _format_json_schema_to_gapic_fixed(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Simplified version for our fix."""
    return _get_items_from_schema_fixed(schema)

# Apply the fix by monkey patching
def apply_gemini_schema_fix():
    """Apply the fix to LangChain Google GenAI."""
    try:
        import langchain_google_genai._function_utils as func_utils
        func_utils._get_properties_from_schema = _get_properties_from_schema_fixed
        func_utils._get_properties_from_schema_any = _get_properties_from_schema_any_fixed
        func_utils._get_items_from_schema_any = _get_items_from_schema_any_fixed
        func_utils._get_items_from_schema = _get_items_from_schema_fixed
        print("✅ Applied Gemini schema fix successfully!")
        return True
    except ImportError as e:
        print(f"❌ Failed to apply fix: {e}")
        return False

if __name__ == "__main__":
    apply_gemini_schema_fix()
