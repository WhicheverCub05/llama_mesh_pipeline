def filter_gemma3(input_text: str) -> str:
    """Remove Gemma3 artifacts from output."""
    input_text = input_text.strip()
    if not input_text:
        return ""

    if ":" in input_text:
        return input_text[input_text.find(":")+1:].strip()
    return input_text


def filter_llama3(input_text: str) -> str:
    """Llama3 passthrough - return input as-is."""
    return input_text.strip() if input_text else ""


filter_list = {
    "gemma3:1b": filter_gemma3,
    "gemma3:4b": filter_gemma3,
    "llama3.1-8b-mesh": filter_llama3,
    "llama3.1-8b": filter_llama3
}


def filter_model_output(input: str, model: str) -> str:
    """Filter model output based on model type."""
    if not input:
        return ""

    filter_func = filter_list.get(model)
    if filter_func:
        result = filter_func(input)
        return result if isinstance(result, str) and result else ""
    return input