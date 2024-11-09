import unittest
import time


def compress_string(string: str) -> str:
    compressed = []
    counter = 0

    for i in range(len(string)):
        if string[i-1] != string[i]:
            compressed.append(string[i-1] + str(counter))
            counter = 0
        counter += 1
    
    if counter: 
        compressed.append(string[-1] + str(counter))
    
    return min(string, "".join(compressed), key=len)


class Test(unittest.TestCase):
    test_cases = [
        ("aabcccccaaa", "a2b1c5a3"),
        ("abcdef", "abcdef"),
        ("aabb", "aabb"),
        ("aaa", "a3"),
        ("a", "a"),
        ("", ""),
    ]
    testable_functions = [
        compress_string,
    ]

    def test_string_compression(self):
        for function in self.testable_functions:
            start = time.perf_counter()
            for _ in range(1000):
                for test_string, expected in self.test_cases:
                    assert function(test_string) == expected
            duration = time.perf_counter() - start
            print(f"{function.__name__} {duration * 1000:.1f}ms")


if __name__ == "__main__":
    unittest.main()
