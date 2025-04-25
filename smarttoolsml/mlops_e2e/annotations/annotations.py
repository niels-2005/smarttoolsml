from ensure import ensure_annotations 

@ensure_annotations 
def get_product(x: int, y: int) -> int:
    return x * y 

# wirft error wenn Annotations nicht gematcht sind