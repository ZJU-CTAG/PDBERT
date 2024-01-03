import re

def clean_signature_line(code: str) -> str:
    code = re.sub(r'( |\t|\n)+', ' ', code)
    return code


def convert_func_signature_to_one_line(code_path=None, code=None):
    if code_path is not None:
        with open(code_path, 'r') as f:
            text = f.read()
    else:
        text = code

    left_bracket_first_idx = text.find('{')
    signature_text = text[:left_bracket_first_idx]
    signature_text = clean_signature_line(signature_text).strip()
    text = signature_text + '\n' + text[left_bracket_first_idx:]

    return text