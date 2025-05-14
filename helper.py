# coding: utf-8
# Copyright IsNoobGrammer and aloobun, 2025


import random
import math


SEPARATOR = "#$TinyCompany@&" 

def parse_distribution(distribution_dict):
    """Parses and validates the distribution dictionary."""
    parsed_dist = []
    total_prob = 0.0
    overall_min_len = float('inf')
    overall_max_len = 0

    for range_str, probability in distribution_dict.items():
        try:
            min_val_str, max_val_str = range_str.split('-')
            min_val = int(min_val_str)
            max_val = int(max_val_str)

            if not (0.0 <= probability <= 1.0):
                raise ValueError(f"Probability must be between 0.0 and 1.0, got {probability} for range '{range_str}'")
            if min_val <= 0:
                 raise ValueError(f"Minimum length must be > 0, got {min_val} for range '{range_str}'")
            if min_val > max_val:
                raise ValueError(f"Minimum length ({min_val}) cannot be greater than maximum length ({max_val}) for range '{range_str}'")

            parsed_dist.append({
                "range": (min_val, max_val),
                "probability": probability
            })
            total_prob += probability
            overall_min_len = min(overall_min_len, min_val)
            overall_max_len = max(overall_max_len, max_val)

        except ValueError as e:
            raise ValueError(f"Invalid range format or value in '{range_str}': {e}") from e
        except Exception as e:
            raise ValueError(f"Error parsing distribution entry '{range_str}': {probability}. Details: {e}") from e

    if not math.isclose(total_prob, 1.0, abs_tol=1e-9):
        raise ValueError(f"Probabilities in distribution must sum to 1.0, but they sum to {total_prob}")
    if not parsed_dist:
        raise ValueError("Distribution dictionary cannot be empty.")

    return parsed_dist, overall_min_len, overall_max_len

def generate_chunk_lengths_from_distribution(text_length, parsed_distribution, overall_min_len):
    """Generates chunk lengths based on the provided parsed distribution."""
    lengths = []
    current_total_length = 0
    population_ranges = [item["range"] for item in parsed_distribution]
    weights = [item["probability"] for item in parsed_distribution]

    while current_total_length < text_length:
        remaining_length = text_length - current_total_length
        if remaining_length <= 0: break # Done

        if remaining_length < overall_min_len:
             chunk_len = remaining_length
             if chunk_len > 0: lengths.append(chunk_len)
             break

        chosen_range = random.choices(population_ranges, weights=weights, k=1)[0]
        min_r, max_r = chosen_range
        max_possible_in_range = min(max_r, remaining_length)
        min_possible_in_range = min(min_r, remaining_length)

        if max_possible_in_range < min_possible_in_range:
             chunk_len = max(1, remaining_length)
        else:
            chunk_len = random.randint(min_possible_in_range, max_possible_in_range)

        if chunk_len <= 0: chunk_len = max(1, remaining_length)

        lengths.append(chunk_len)
        current_total_length += chunk_len
    return lengths

def augment_text_with_distribution(text, distribution_dict, separator=SEPARATOR):
    """
    Inserts a separator into text based on chunk lengths following the provided distribution.

    Args:
        text (str): The input text string.
        distribution_dict (dict): A dictionary defining chunk length ranges and their probabilities
                                  (e.g., {"1-4": 0.15, "5-15": 0.70, "16-25": 0.15}).
                                  Probabilities must sum to 1.0. Keys are "min-max" strings.
                                  Min length must be > 0.
        separator (str, optional): The separator string to insert. Defaults to SEPARATOR.

    Returns:
        str: The text with the separator inserted between chunks.

    Raises:
        ValueError: If the distribution_dict is invalid.
    """
    if not text: return ""
    parsed_distribution, overall_min_len, _ = parse_distribution(distribution_dict)
    text_length = len(text)
    chunk_lengths = generate_chunk_lengths_from_distribution(text_length, parsed_distribution, overall_min_len)

    augmented_parts = []
    current_pos = 0
    for i, length in enumerate(chunk_lengths):
        end_pos = min(current_pos + length, text_length)
        chunk = text[current_pos : end_pos]
        if chunk:
            augmented_parts.append(chunk)
            if i < len(chunk_lengths) - 1:
                augmented_parts.append(separator)
        current_pos = end_pos
        if current_pos >= text_length: break
    return "".join(augmented_parts)