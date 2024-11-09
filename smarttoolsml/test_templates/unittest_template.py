import time
import unittest
from collections import defaultdict


# O(n), O(1)
def is_unique_chars_algorithmic(string: str) -> bool:
    if len(string) == 1:
        return True

    if len(string) > 128:
        return False

    char_set = [False] * 128
    for char in string:
        val = ord(char)
        if char_set[val]:
            return False
        char_set[val] = True
    return True


# O(n), O(n)
def is_unique_chars_pythonic(string: str) -> bool:
    if len(string) == 1:
        return True

    return len(set(string)) == len(string)


# O(n), O(n)
def is_unique_chars_using_dictionary(string: str) -> bool:
    if len(string) == 1:
        return True

    character_counts = {}

    for char in string:
        if char in character_counts:
            return False
        character_counts[char] = 1
    return True


class Test(unittest.TestCase):
    test_cases = [
        ("a", True),
        ("abcd", True),
        ("aabbccd", False),
        ("mevst", True),
        ("mmeeeeoooowww", False),
    ]

    test_functions = [
        is_unique_chars_algorithmic,
        is_unique_chars_pythonic,
        is_unique_chars_using_dictionary,
    ]

    def test_is_unique_chars(self):
        num_runs = 100000
        functions_runtimes = defaultdict(float)

        for _ in range(num_runs):
            for text, expected in self.test_cases:
                for is_unique_chars in self.test_functions:
                    start = time.perf_counter()

                    assert (
                        is_unique_chars(text) == expected
                    ), f"{is_unique_chars.__name__} failed for value: {text}"

                    functions_runtimes[is_unique_chars.__name__] += (
                        time.perf_counter() - start
                    ) * 1000

        print(f"\n{num_runs} runs")
        for function_name, runtime in functions_runtimes.items():
            print(f"{function_name}: {runtime:.1f}ms")


if __name__ == "__main__":
    unittest.main()
