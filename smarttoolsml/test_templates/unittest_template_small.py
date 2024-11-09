def check_permutation_by_sort(s1: str, s2: str) -> bool:
    if len(s1) != len(s2):
        return False

    s1, s2 = sorted(s1), sorted(s2)

    for i in range(len(s1)):
        if s1[i] != s2[i]:
            return False
    return True


def check_permutation_by_count(s1: str, s2: str) -> bool:
    if len(s1) != len(s2):
        return False

    # assuming ascii
    counter = [0] * 128

    for char in s1:
        counter[ord(char)] += 1

    for char in s2:
        if counter[ord(char)] == 0:
            return False
        counter[ord(char)] -= 1
    return True


import unittest


class Test(unittest.TestCase):
    test_cases = [
        ("dog", "god", True),
        ("dog ", "god ", True),
        ("dog", "cat", False),
        ("DOG", "god", False),
    ]

    test_functions = [check_permutation_by_count, check_permutation_by_sort]

    def test_permutation(self):
        for check_permutation in self.test_functions:
            for s1, s2, expected in self.test_cases:
                assert check_permutation(s1, s2) == expected


if __name__ == "__main__":
    unittest.main()
