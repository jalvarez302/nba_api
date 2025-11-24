import re
import warnings
from nba_api.stats.library.data import players, wnba_players
from nba_api.stats.library.data import (
    player_index_id,
    player_index_full_name,
    player_index_first_name,
    player_index_last_name,
    player_index_is_active,
)
import unicodedata

# Security: Maximum pattern length to prevent ReDoS attacks
MAX_PATTERN_LENGTH = 200


def _find_players(regex_pattern, row_id, players=players, safe_search=False):
    """
    Find players matching the regex pattern.

    Args:
        regex_pattern: The regex pattern to search for
        row_id: The index of the player attribute to search
        players: The list of players to search
        safe_search: If True, escape the pattern to prevent ReDoS attacks.
                     Use this when the pattern comes from untrusted user input.

    Returns:
        List of matching player dictionaries
    """
    # Security: Limit pattern length to prevent ReDoS
    if len(regex_pattern) > MAX_PATTERN_LENGTH:
        warnings.warn(
            f"Pattern length exceeds {MAX_PATTERN_LENGTH} characters. Truncating for security.",
            UserWarning
        )
        regex_pattern = regex_pattern[:MAX_PATTERN_LENGTH]

    # Security: Optionally escape the pattern to prevent ReDoS
    if safe_search:
        regex_pattern = re.escape(regex_pattern)

    players_found = []
    try:
        for player in players:
            if re.search(_strip_accents(regex_pattern), _strip_accents(str(player[row_id])), flags=re.I):
                players_found.append(_get_player_dict(player))
    except re.error as e:
        raise ValueError(f"Invalid regex pattern: {e}")
    return players_found


def _strip_accents(inputstr: str) -> str:
    """
    Normalize and remove accents from string.
    """
    # Normalize to decomposed form
    normalizedstr = unicodedata.normalize('NFD', inputstr)
    # Filter out accents (Mn = Mark, Nonspacing category)
    return ''.join(charx for charx in normalizedstr if unicodedata.category(charx) != 'Mn')


def _find_player_by_id(player_id, players=players):
    regex_pattern = "^{}$".format(player_id)
    players_list = _find_players(regex_pattern, player_index_id, players=players)
    if len(players_list) > 1:
        raise Exception("Found more than 1 id")
    elif not players_list:
        return None
    else:
        return players_list[0]


def _get_players(players=players):
    players_list = []
    for player in players:
        players_list.append(_get_player_dict(player))
    return players_list


def _get_active_players(players=players):
    players_list = []
    for player in players:
        if player[player_index_is_active]:
            players_list.append(_get_player_dict(player))
    return players_list


def _get_inactive_players(players=players):
    players_list = []
    for player in players:
        if not player[player_index_is_active]:
            players_list.append(_get_player_dict(player))
    return players_list


def _get_player_dict(player_row):
    return {
        "id": player_row[player_index_id],
        "full_name": player_row[player_index_full_name],
        "first_name": player_row[player_index_first_name],
        "last_name": player_row[player_index_last_name],
        "is_active": player_row[player_index_is_active],
    }


def find_players_by_full_name(regex_pattern, safe_search=False):
    """
    Find players by full name using regex pattern.

    Args:
        regex_pattern: The pattern to search for
        safe_search: If True, escape the pattern for literal matching (recommended for user input)
    """
    return _find_players(regex_pattern, player_index_full_name, safe_search=safe_search)


def find_players_by_first_name(regex_pattern, safe_search=False):
    """
    Find players by first name using regex pattern.

    Args:
        regex_pattern: The pattern to search for
        safe_search: If True, escape the pattern for literal matching (recommended for user input)
    """
    return _find_players(regex_pattern, player_index_first_name, safe_search=safe_search)


def find_players_by_last_name(regex_pattern, safe_search=False):
    """
    Find players by last name using regex pattern.

    Args:
        regex_pattern: The pattern to search for
        safe_search: If True, escape the pattern for literal matching (recommended for user input)
    """
    return _find_players(regex_pattern, player_index_last_name, safe_search=safe_search)


def find_player_by_id(player_id):
    return _find_player_by_id(player_id)


def get_players():
    return _get_players()


def get_active_players():
    return _get_active_players()


def get_inactive_players():
    return _get_inactive_players()


def find_wnba_players_by_full_name(regex_pattern, safe_search=False):
    """
    Find WNBA players by full name using regex pattern.

    Args:
        regex_pattern: The pattern to search for
        safe_search: If True, escape the pattern for literal matching (recommended for user input)
    """
    return _find_players(regex_pattern, player_index_full_name, players=wnba_players, safe_search=safe_search)


def find_wnba_players_by_first_name(regex_pattern, safe_search=False):
    """
    Find WNBA players by first name using regex pattern.

    Args:
        regex_pattern: The pattern to search for
        safe_search: If True, escape the pattern for literal matching (recommended for user input)
    """
    return _find_players(regex_pattern, player_index_first_name, players=wnba_players, safe_search=safe_search)


def find_wnba_players_by_last_name(regex_pattern, safe_search=False):
    """
    Find WNBA players by last name using regex pattern.

    Args:
        regex_pattern: The pattern to search for
        safe_search: If True, escape the pattern for literal matching (recommended for user input)
    """
    return _find_players(regex_pattern, player_index_last_name, players=wnba_players, safe_search=safe_search)


def find_wnba_player_by_id(player_id):
    return _find_player_by_id(player_id, players=wnba_players)


def get_wnba_players():
    return _get_players(players=wnba_players)


def get_wnba_active_players():
    return _get_active_players(players=wnba_players)


def get_wnba_inactive_players():
    return _get_inactive_players(players=wnba_players)
