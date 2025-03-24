def extract_response_from_thinking(completion):
    """
    从思维链输出中提取最终回答部分
    Args:
        completion: 可能包含思维链的完整回答字符串
    Returns:
        提取出的回答部分
    """
    if "</think>" in completion:
        response = completion.split("</think>")[-1]
    else:
        response = completion
    return response
