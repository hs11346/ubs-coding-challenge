import re
import numpy as np

def extract_row_blocks(mat: np.ndarray):
    """
    For each consecutive-row block in `mat`, return a mapping
    blob_bytes -> list of (start, end) positions where it occurs.
    """
    n_rows, _ = mat.shape
    blocks = {}
    for start in range(n_rows):
        for end in range(start+1, n_rows+1):
            blob = mat[start:end, :].tobytes()
            blocks.setdefault(blob, []).append((start, end))
    return blocks

def find_maximal_unique_common_rows(
    list1,
    list2
) :
    # 1) Build blob→positions maps for every matrix
    maps1 = [extract_row_blocks(m) for m in list1]
    maps2 = [extract_row_blocks(m) for m in list2]
    
    # 2) Intersect blobs across list1
    common_blobs = set(maps1[0].keys())
    for mp in maps1[1:]:
        common_blobs &= set(mp.keys())
    if not common_blobs:
        return []
    
    # 3) Subtract those that are common to *every* list2 matrix
    if maps2:
        blobs2 = set(maps2[0].keys())
        for mp in maps2[1:]:
            blobs2 &= set(mp.keys())
        common_blobs -= blobs2
        if not common_blobs:
            return []
    
    # 4) For each remaining blob, gather all its (start,end) positions in the *reference* matrix
    ref_map = maps1[0]
    candidates = []  # (start, end, blob)
    for blob in common_blobs:
        for (s,e) in ref_map[blob]:
            candidates.append((s, e, blob))
    
    # 5) Greedy maximal filtering (by height descending)
    candidates.sort(key=lambda x: x[1]-x[0], reverse=True)
    maximal = []
    for s,e,blob in candidates:
        # drop if fully contained in any already accepted slice
        if any(s0 <= s and e <= e0 for s0,e0,_ in maximal):
            continue
        maximal.append((s,e,blob))
    
    # 6) Reconstruct the actual arrays from the reference matrix
    dtype = list1[0].dtype
    n_cols = list1[0].shape[1]
    out = []
    for s,e,blob in maximal:
        arr = np.frombuffer(blob, dtype=dtype).reshape((e-s, n_cols))
        out.append(arr)
    return out

def find_minimal_unique_common_rows(list1, list2=None):
    """
    Extract the *smallest* unique common row-blocks present in all matrices of list1
    but not in any matrix of list2. Returns list of np.ndarray blocks.
    """
    # Build block maps
    maps1 = [extract_row_blocks(m) for m in list1]
    maps2 = [extract_row_blocks(m) for m in list2] if list2 else []

    # Common blobs in list1
    common = set(maps1[0].keys())
    for mp in maps1[1:]:
        common &= set(mp.keys())
    # Exclude ones also in all of list2
    if maps2:
        blobs2 = set(maps2[0].keys())
        for mp in maps2[1:]:
            blobs2 &= set(mp.keys())
        common -= blobs2
    if not common:
        return []

    # Collect positions from reference
    ref = maps1[0]
    candidates = [(s, e, b) for b in common for (s, e) in ref[b]]

    # Keep smallest non-overlapping blocks first
    candidates.sort(key=lambda x: x[1] - x[0])  # ascending length
    chosen = []
    for s, e, blob in candidates:
        # skip if this block fully contains any already chosen smaller one
        if any(s <= s0 and e0 <= e for s0, e0, _ in chosen):
            continue
        chosen.append((s, e, blob))

    # Reconstruct arrays
    dtype = list1[0].dtype
    n_cols = list1[0].shape[1]
    return [np.frombuffer(blob, dtype=dtype).reshape((e-s, n_cols)) for s, e, blob in chosen]

def find_minimal_unique_patterns(
    list1: list[np.ndarray],
    list2: list[np.ndarray]
) -> list[np.ndarray]:
    """
    Find minimal consecutive-row blocks that appear in every matrix of `list2`
    but in none of the matrices in `list1`.
    Returns a list of unique numpy arrays from the first matrix in list2,
    choosing the smallest non-overlapping blocks.
    """
    # 1) Build blob->positions maps for list2 and list1
    maps2 = [extract_row_blocks(m) for m in list2]
    maps1 = [extract_row_blocks(m) for m in list1]

    # 2) Intersect blobs across all matrices in list2
    common2 = set(maps2[0].keys())
    for mp in maps2[1:]:
        common2 &= set(mp.keys())
    if not common2:
        return []

    # 3) Remove any blobs that appear in any matrix of list1
    if maps1:
        blobs1 = set().union(*(set(mp.keys()) for mp in maps1))
        unique_blobs = common2 - blobs1
    else:
        unique_blobs = common2

    if not unique_blobs:
        return []

    # 4) Gather candidate positions from reference matrix list2[0]
    ref_map = maps2[0]
    candidates = [(s, e, blob)
                  for blob in unique_blobs
                  for s, e in ref_map.get(blob, [])]

    # 5) Greedy minimal filtering by block height (ascending)
    candidates.sort(key=lambda x: x[1] - x[0])  # shortest first
    minimal = []
    for s, e, blob in candidates:
        # Skip if this block fully contains any already accepted block
        if any(s <= s0 and e0 <= e for s0, e0, _ in minimal):
            continue
        minimal.append((s, e, blob))

    # 6) Reconstruct numpy arrays from blobs
    dtype = list2[0].dtype
    n_cols = list2[0].shape[1]
    result = []
    for s, e, blob in minimal:
        arr = np.frombuffer(blob, dtype=dtype).reshape((e - s, n_cols))
        result.append(arr)

    return result

def max_padding_length(list1, list2, pad_token="★"):
    """
    Determines the maximum number of padding characters added to any string when padding list1 and list2.

    Args:
        list1 (list of str): First list of original strings.
        list2 (list of str): Second list of original strings.
        pad_token (str): The token used for padding (default: "<PAD>").

    Returns:
        int: The maximum count of padding characters added to any single string.
    """
    # Get the padded versions
    padded_list1, padded_list2 = pad_strings(list1, list2, pad_token)

    # Compute padding lengths for each string
    padding_lengths = []
    for original, padded in zip(list1, padded_list1):
        padding_lengths.append(len(padded) - len(original))
    

    return max(padding_lengths) if padding_lengths else 0

def pad_strings(list1, list2, pad_token="★"):
    """
    Pads each string in list1 and list2 to the length of the longest string in either list,
    using the specified pad_token. The pad_token is repeated or truncated as needed.

    Args:
        list1 (list of str): First list of strings to pad.
        list2 (list of str): Second list of strings to pad.
        pad_token (str): The token to use for padding (default: "<PAD>").

    Returns:
        tuple: Two lists of padded strings (padded_list1, padded_list2).
    """
    # Combine both lists to find the maximum string length
    combined = list1 + list2
    if not combined:
        return list1, list2

    max_length = max(len(s) for s in combined)

    def pad(s):
        # Calculate how many characters we need to add
        needed = max_length - len(s)
        if needed <= 0:
            return s
        # Repeat the token and truncate to exactly the needed length
        pad_sequence = (pad_token * needed)[:needed]
        return s + pad_sequence

    # Apply padding to both lists
    padded_list1 = [pad(s) for s in list1]
    padded_list2 = [pad(s) for s in list2]

    return padded_list1, padded_list2

def matrix_to_string(matrix: np.ndarray, index: list, has_pos) -> str:
    """
    Invert string_to_matrix_extended:
      - If any of the unique-character one-hot bits is set, return that exact character.
      - Otherwise, return a placeholder token for the character class.
    
    Arguments:
    - matrix: 2D np.ndarray, each row is
        [is_lower, is_upper, is_word, is_digit, is_special, is_whitespace, is_pad,
         *unique_char_onehot (len=index)*, char_pos]
    - index: list of all unique characters in the same order used by string_to_matrix_extended
    
    Returns:
    - A single string where each “character” is either the recovered character (if present)
      or a class token like "<LOWERCASE>", "<UPPERCASE>", "<DIGIT>", "<SPECIAL>", "<SPACE>", "<PAD>".
    """
    rows, cols = matrix.shape
    # There are always 7 class flags + 1 char_pos at the end.
    # If cols > 8, we have unique_char_onehot bits between them.
    has_index = cols > 8
    base_flags = 7
    tokens = []
    
    if has_index:
        if has_pos:
            min_pos = matrix[:, -1].min()
            if min_pos > 0:
                tokens.append("^.{" + str(min_pos) + "}")
            else:
                tokens.append("^")
            for row in matrix:
                flags = row[base_flags:-1].astype(int)
                
                for idx, val in enumerate(flags):
                    if val != 0:
                        tokens.append(re.escape(index[idx]))
        else:
            tokens.append(".*")
            for row in matrix:
                flags = row[base_flags:].astype(int)
                for idx, val in enumerate(flags):
                    if val != 0:
                        tokens.append(re.escape(index[idx]))
        
    else:
        if has_pos:
            min_pos = matrix[:, -1].min()
            if min_pos > 0:
                tokens.append("^.{" + str(min_pos) + "}")
            else:
                tokens.append("^")
        else:
            tokens.append(".*")
        for row in matrix:
            flags = row[:base_flags].astype(int)
            if flags[0]:
                tokens.append("[a-z]")
            elif flags[1]:
                tokens.append("[A-Z]")
            elif flags[2]:
                tokens.append("\w")
            elif flags[3]:
                tokens.append("\d")
            elif flags[4]:
                tokens.append("[^\w\d]")
            elif flags[5]:
                tokens.append(" ")
            elif flags[6]:
                pass
            else:
                # should not happen if your encoder asserts
                raise Exception("Unknown behaviour")
        
        
        
    
    # Join without delimiters: you’ll get a string of tokens.
    # If you need spaces or separators, adjust as you like.
    return ''.join(tokens) + ".*"

def matches_pattern(string, pattern):
    """
    Tests if the entire string matches the regex pattern.

    Args:
        string (str): The string to be tested.
        pattern (str): The regular expression pattern.

    Returns:
        bool: True if the string matches the pattern, False otherwise.
    """
    # Use re.fullmatch to ensure the entire string is checked.
    return bool(re.fullmatch(pattern, string))


def test_pattern(valid_strings, pattern, verbose = True):
    """
    Tester function that uses the provided output function (out_func) to generate a regex pattern.
    It then verifies that all valid strings match the regex fully and that all invalid strings do not.
    
    Args:
        valid_strings (list): List of strings expected to match the regex.
        invalid_strings (list): List of strings that should not match the regex.
        out_func (function): A function that takes (valid_strings, invalid_strings) as parameters
                             and returns a regex string.
    
    Returns:
        bool: True if the regex passes all tests, False otherwise.
    """
    # Generate the Gree Expression (a regex pattern, per the challenge)

    if verbose:
        print(f"Generated Regex Pattern: {pattern}\n")
    
    passed = True

    # Test valid strings
    for s in valid_strings:
        if not matches_pattern(s, pattern):
            if verbose:
                print(f"FAIL: Valid string '{s}' did not match!")
            passed = False
        else:
            if verbose:
                print(f"PASS: Valid string '{s}' matched.")

 
    return passed

def test_pattern_all_negative(valid_strings, pattern, verbose = True):
 
    # Generate the Gree Expression (a regex pattern, per the challenge)

    if verbose:
        print(f"Generated Regex Pattern: {pattern}\n")
    
    passed = True

    # Test valid strings
    for s in valid_strings:
        if  matches_pattern(s, pattern):
            if verbose:
                print(f"FAIL: Valid string '{s}' did not match!")
            passed = False
        else:
            if verbose:
                print(f"PASS: Valid string '{s}' matched.")

 
    return passed

import itertools
from itertools import permutations

def build_sorted_patterns(candidates):
    patterns = []
    # r = size of combination: 1 through len(candidates)

    for r in range(1, len(candidates) + 1):
        for combo in itertools.combinations(candidates, r):
            # build the lookahead sequence for this combo
            if len(combo) > 1:
                lookaheads = "".join(f"(?={i})" for i in combo)
                pattern = f"^{lookaheads}.*$"
                patterns.append(pattern)
            else:
                if combo[0][0] == '^':
                    patterns.append(f"{combo[0]}$")
                else:
                    patterns.append(f"^{combo[0]}$")
    
    # combine into one lookahead 
    for r in range(2, len(candidates) + 1):
        for combo in itertools.combinations(candidates, r):
            # build the lookahead sequence for this combo
            start_patterns = [i[:-2] for i in combo if i[0] == "^"]
            not_start_patterns = [i for i in combo if i[0] != "^"]
            if len(start_patterns) <= 1:
                 for perm in permutations(not_start_patterns):
                    lookaheads = "".join(start_patterns + list(perm))
                    # ensure we have exactly one leading ^
                    if lookaheads.startswith("^"):
                        pattern = f"{lookaheads}$"
                    else:
                        pattern = f"^{lookaheads}$"
                    patterns.append(pattern)
    # sort ascending by string length
    patterns.sort(key=len)
    return patterns
def invert_regex(pattern: str) -> str:
    """
    Given a regex pattern, returns a new pattern that matches
    exactly those strings the original would reject.

    This works by ensuring the original is treated as a full‐string
    match (adding ^…$ if missing), stripping those anchors off,
    then wrapping it in a negative lookahead.

    Example:
        invert_regex("^x.*")  -> "^(?!x.*).*$"
    """
    # Ensure full‐string matching
    if not pattern.startswith("^"):
        pattern = "^" + pattern
    if not pattern.endswith("$"):
        pattern = pattern + "$"

    # Strip the anchors
    inner = pattern[1:-1]

    # Wrap in negative lookahead
    return f"^(?!{inner}).*$"


def collapse_wildcards(pattern: str) -> str:
    """
    Replace any sequence of two or more '.*' in the regex pattern
    with a single '.*'.
    """
    # (?:\.\*){2,}  → non-capturing group of '.*', repeated 2 or more times
    return re.sub(r'(?:\.\*){2,}', '.*', pattern)

def collapse_char_classes(pattern: str) -> str:
    """
    Replace all occurrences of [a-z] or [A-Z] with \w in the given pattern.
    """
    # Use '\\w' so that the replacement is a literal backslash+w
    return re.sub(r'\[(?:a-z|A-Z)\]', r'\\w', pattern)

def shorten_classes(pattern: str) -> str:
    """
    Collapse runs of identical regex character-class tokens into one with a quantifier.
    E.g. "\\w\\w\\w" → "\\w{3}".
    """
    # This regex looks for one of \w, \d, \s, \W, \D, \S
    # captured in group 1, followed by one or more repeats of that same group.
    token_re = re.compile(r'(\\[wWsSdD])(?:\1)+')

    def _repl(match: re.Match) -> str:
        tok = match.group(1)                      # e.g. "\w"
        total = len(match.group(0)) // len(tok)   # how many repeats
        return f'{tok}{{{total}}}'

    return token_re.sub(_repl, pattern)

def get_unique_chars(strings):
    """
    Args:
        strings (list of str): List of input strings.
    Returns:
        list of str: Sorted list of unique characters.
    """
    unique_chars = set()
    for s in strings:
        unique_chars.update(s)
    return sorted(unique_chars)

def char_to_vector_extended(ch, index, char_pos, add_index) -> list:
    # Check that the input is a single character
    if len(ch) != 1:
        raise ValueError("Input must be a single character string")

    # Determine membership for each group
    is_lowercase = ch.islower()       # Lowercase letter
    is_uppercase = ch.isupper()       # Uppercase letter
    is_word = ch.isalpha() or ch == "_"
    is_digit     = ch.isdigit()        # Numeric (digit)
    # Special character: a character that's not a letter or digit.
    is_special   = (not ch.isalnum()) and not (ch == " ") and not (ch == '_') and not (ch == '★')
    is_whitespace = (ch == " ")
    is_pad = (ch == '★')
    vec = [is_lowercase, is_uppercase, is_word, is_digit, is_special, is_whitespace,is_pad]
    assert any(vec) == True, f"Uncaught character: {ch} {ch.islower()}"

    # Return the vector in the specified order:
    # [lowercase, uppercase, numeric, special, whitespace, pad, all unique characters]
    if add_index:
        return  [int(i) for i in vec] + [int(ch == i) for i in index] + [char_pos]
    else:
        return  [int(i) for i in vec] + [char_pos]

def string_to_matrix_extended(x : str, index, add) -> list:
    output = []
    for char_pos, char in enumerate(x):
        output.append(char_to_vector_extended(char, index, char_pos, add))
    return output


def generate_gree_expression(valid_strings_raw, invalid_strings_raw):
    ch_index = get_unique_chars(valid_strings_raw + invalid_strings_raw)
    max_padding = max_padding_length(valid_strings_raw, invalid_strings_raw)
    valid_strings, invalid_strings = pad_strings(valid_strings_raw, invalid_strings_raw)

    valid_strings_extended_matrix_list = []
    invalid_strings_extended_matrix_list = []

    for i in valid_strings:
        valid_strings_extended_matrix_list.append(np.array(string_to_matrix_extended(i, ch_index, True)))
    for i in invalid_strings:
        invalid_strings_extended_matrix_list.append(np.array(string_to_matrix_extended(i, ch_index, True)))

    valid_strings_matrix_list = []
    invalid_strings_matrix_list = []
    for i in valid_strings:
        valid_strings_matrix_list.append(np.array(string_to_matrix_extended(i, ch_index, False)))
    for i in invalid_strings:
        invalid_strings_matrix_list.append(np.array(string_to_matrix_extended(i, ch_index, False)))
    
    candidates = []
   
    # Positive membership
    # 1. Include position + actual character (character at position match)
    full_subset_in_valid_not_all_in_invalid = find_maximal_unique_common_rows(valid_strings_extended_matrix_list, invalid_strings_extended_matrix_list)
    for i in range(len(full_subset_in_valid_not_all_in_invalid)):
        candidates.append(matrix_to_string(full_subset_in_valid_not_all_in_invalid[i], ch_index, True))

    # 2. Include actual character  (substring match regardless of position)
    second_subset_in_valid_not_all_in_invalid = find_maximal_unique_common_rows([i[:, :-1] for i in valid_strings_extended_matrix_list], [i[:, :-1] for i in invalid_strings_extended_matrix_list])
    for i in range(len(second_subset_in_valid_not_all_in_invalid)):
        candidates.append(matrix_to_string(second_subset_in_valid_not_all_in_invalid[i], ch_index, False))
    # 3. Include character class membership  + position (character class at position match)
    third_subset_in_valid_not_all_in_invalid = find_maximal_unique_common_rows([i for i in valid_strings_matrix_list], [i for i in invalid_strings_matrix_list])
    for i in range(len(third_subset_in_valid_not_all_in_invalid)):
        candidates.append(matrix_to_string(third_subset_in_valid_not_all_in_invalid[i], ch_index, True))

 
    # 4. Include character class membership (character class match)
    fourth_subset_not_all_in_valid_in_invalid = find_maximal_unique_common_rows([i[:, :-1] for i in invalid_strings_matrix_list], [i[:, :-1] for i in valid_strings_matrix_list])
    for i in range(len(fourth_subset_not_all_in_valid_in_invalid)):
        candidates.append(matrix_to_string(fourth_subset_not_all_in_valid_in_invalid[i], ch_index, False))

    candidates = set([i for i in candidates if test_pattern(valid_strings_raw, f"^{i}.*$", verbose=False) == True and test_pattern(invalid_strings_raw, f"^{i}.*$", verbose=False) == False  ])
    # Negative membership (absolute condition)
    # 1. Include position + actual character (character at position match)
    negative_candidates = []
    tmp = find_minimal_unique_patterns(valid_strings_extended_matrix_list, invalid_strings_extended_matrix_list)
    for i in range(len(tmp)):
        negative_candidates.append(matrix_to_string(tmp[i], ch_index, True))

    # 2. Include actual character  (substring match regardless of position)
    tmp = find_minimal_unique_patterns([i[:, :-1] for i in valid_strings_extended_matrix_list], [i[:, :-1] for i in invalid_strings_extended_matrix_list])
    for i in range(len(tmp)):
        negative_candidates.append(matrix_to_string(tmp[i], ch_index, False))
    # 3. Include character class membership  + position (character class at position match)
    tmp = find_minimal_unique_patterns([i for i in valid_strings_matrix_list], [i for i in invalid_strings_matrix_list])
    for i in range(len(tmp)):
        negative_candidates.append(matrix_to_string(tmp[i], ch_index, True))

 
    # 4. Include character class membership (character class match)
    tmp = find_minimal_unique_patterns([i[:, :-1] for i in valid_strings_matrix_list], [i[:, :-1] for i in invalid_strings_matrix_list])
    for i in range(len(tmp)):
        negative_candidates.append(matrix_to_string(tmp[i], ch_index, False))
    
    negative_candidates = [i for i in negative_candidates if test_pattern_all_negative(valid_strings_raw, i, verbose=False) == True and test_pattern(invalid_strings_raw, i, verbose=False) == True]
    #candidates += negative_candidates
    patterns = build_sorted_patterns(candidates)  + [invert_regex(i) for i in negative_candidates]
    all_combo =  sorted([shorten_classes(collapse_wildcards(i)) for i in patterns] + [shorten_classes(collapse_wildcards(collapse_char_classes(i))) for i in patterns], key=len)
    for combo in all_combo:
        if test_pattern(valid_strings_raw, combo, False) and test_pattern_all_negative(invalid_strings_raw, combo, False):
            return combo
    return "" # fail
 
