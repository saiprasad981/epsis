"""
geocode.py — Location name -> candidate (lat, lon) matches
=============================================================
Handles villages, districts, cities, towns — anything OpenStreetMap has.

Two backends, tried in order:
  1. Photon (photon.komoot.io) — free, OSM-based, no API key, no strict
     header requirements. Primary choice.
  2. Nominatim (nominatim.openstreetmap.org) — fallback if Photon returns
     nothing. Nominatim's bot-protection layer sometimes returns 403 even
     with a valid User-Agent, depending on network/IP — that's a
     server-side block we can't fully control from our side, which is why
     it's the fallback rather than the primary.

Returns a LIST of candidates (not just the first hit) with a place_type
field (village/town/city/state_district/etc.), because place names repeat
across different states/districts in India — the app should let the user
pick the right one rather than silently guessing.
"""

import requests

PHOTON_URL = "https://photon.komoot.io/api/"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# A default `requests` User-Agent (e.g. "python-requests/2.31.0") is a known
# bot signature that a lot of college/corporate firewalls and proxies block
# outright, regardless of which service you're calling. Using a normal
# browser-like User-Agent avoids that class of block. Nominatim's own policy
# additionally wants a real identifying contact — put yours below.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
NOMINATIM_CONTACT_UA = "EPSIS-BTech-Project/1.0 (contact: your-email@example.com)"  # <-- put a real contact here

COMMON_HEADERS = {
    "User-Agent": BROWSER_USER_AGENT,
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


def geocode_location(query: str, limit: int = 5) -> dict:
    """
    Args:
        query: place name — village, district, city, town, landmark, etc.
        limit: max number of candidate matches to return

    Returns:
        {
            "candidates": [...],
            "errors": [str, ...],       # backend failures encountered
            "any_backend_reachable": bool,  # True if at least one backend
                                             # returned a real (even empty)
                                             # response — tells the caller
                                             # whether "no results" means
                                             # "genuinely no match" or
                                             # "couldn't even check".
        }
    """
    if not query or not query.strip():
        return {"candidates": [], "errors": [], "any_backend_reachable": False}

    errors = []
    any_reachable = False

    photon_candidates, photon_error = _geocode_photon(query, limit)
    if photon_error:
        errors.append(f"Photon: {photon_error}")
    else:
        any_reachable = True  # Photon responded, even if with 0 results
    if photon_candidates:
        return {"candidates": photon_candidates, "errors": errors, "any_backend_reachable": True}

    print("[geocode] Photon returned no matches, trying Nominatim fallback...")
    nominatim_candidates, nominatim_error = _geocode_nominatim(query, limit)
    if nominatim_error:
        errors.append(f"Nominatim: {nominatim_error}")
    else:
        any_reachable = True

    return {
        "candidates": nominatim_candidates,
        "errors": errors,
        "any_backend_reachable": any_reachable,
    }


def _format_photon_label(props: dict) -> str:
    parts = [props.get("name")]
    for key in ("district", "city", "county", "state", "country"):
        val = props.get(key)
        if val and val not in parts:
            parts.append(val)
    return ", ".join(p for p in parts if p)


def _geocode_photon(query: str, limit: int):
    """Returns (candidates: list, error: str|None)."""
    params = {"q": query.strip(), "limit": limit}
    try:
        resp = requests.get(PHOTON_URL, params=params, headers=COMMON_HEADERS, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[geocode] Photon request failed: {e}")
        return [], str(e)

    try:
        data = resp.json()
    except ValueError as e:
        print(f"[geocode] Photon returned non-JSON response: {e}")
        return [], f"invalid response ({e})"

    out = []
    for feature in data.get("features", []):
        try:
            lon, lat = feature["geometry"]["coordinates"]
        except (KeyError, ValueError):
            continue
        props = feature.get("properties", {})
        out.append({
            "lat": lat,
            "lon": lon,
            "display_name": _format_photon_label(props),
            "place_type": props.get("osm_value", "unknown"),
        })

    print(f"[geocode] Photon: query='{query}' -> {len(out)} candidate(s)")
    return out, None


def _geocode_nominatim(query: str, limit: int):
    """Returns (candidates: list, error: str|None)."""
    params = {"q": query.strip(), "format": "json", "limit": limit}
    headers = {**COMMON_HEADERS, "User-Agent": NOMINATIM_CONTACT_UA}
    try:
        resp = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[geocode] Nominatim request failed: {e}")
        return [], str(e)

    try:
        data = resp.json()
    except ValueError as e:
        print(f"[geocode] Nominatim returned non-JSON response: {e}")
        return [], f"invalid response ({e})"

    out = []
    for item in data:
        try:
            out.append({
                "lat": float(item["lat"]),
                "lon": float(item["lon"]),
                "display_name": item.get("display_name", query),
                "place_type": item.get("type", "unknown"),
            })
        except (KeyError, ValueError):
            continue

    print(f"[geocode] Nominatim: query='{query}' -> {len(out)} candidate(s)")
    return out, None
