"""Tracker / embedded-SDK detection from observed network destinations.

Two complementary signals:

1. Known trackers - hosts matching a curated ad/analytics/attribution
   domain list. The list is a small illustrative starting set, not a
   tracker database: it only catches what is on it. Extend KNOWN_TRACKERS
   as you see real traffic.

2. Shared endpoints - domains contacted by several unrelated apps. A
   third-party SDK is embedded in many apps, so an unfamiliar domain that
   shows up across many packages is a strong candidate SDK endpoint even
   when it is on no list. `similar_apps` applies the same idea pairwise
   (Jaccard overlap of contacted-domain sets) to surface apps likely
   sharing an embedded SDK.

Domains are compared at a heuristic base-domain level (last two labels, or
three under common two-letter-TLD second levels like co.uk) so regional
subdomains of one vendor (sdk-us.vendor.com / sdk-eu.vendor.com) count as
one. The heuristic is not the public suffix list; unusual TLDs can group
imperfectly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .schema import EventType, TraceEvent

# Domain suffix -> category. Illustrative starting set, extend as needed.
KNOWN_TRACKERS: dict[str, str] = {
    "doubleclick.net": "ads",
    "googlesyndication.com": "ads",
    "googleadservices.com": "ads",
    "unityads.unity3d.com": "ads",
    "applovin.com": "ads",
    "vungle.com": "ads",
    "chartboost.com": "ads",
    "adcolony.com": "ads",
    "inmobi.com": "ads",
    "mopub.com": "ads",
    "google-analytics.com": "analytics",
    "app-measurement.com": "analytics",
    "mixpanel.com": "analytics",
    "amplitude.com": "analytics",
    "segment.io": "analytics",
    "graph.facebook.com": "social-sdk",
    "appsflyer.com": "attribution",
    "adjust.com": "attribution",
    "branch.io": "attribution",
    "kochava.com": "attribution",
    "singular.net": "attribution",
    "crashlytics.com": "crash-reporting",
    "bugsnag.com": "crash-reporting",
    "sentry.io": "crash-reporting",
}

_IPV4_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_SECOND_LEVEL_LABELS = {"co", "com", "org", "net", "gov", "edu", "ac"}


@dataclass
class SharedDomain:
    domain: str
    packages: list[str]
    known_category: str | None


@dataclass
class SimilarPair:
    package_a: str
    package_b: str
    jaccard: float
    shared_domains: list[str]


def match_known_tracker(host: str) -> tuple[str, str] | None:
    """(matched suffix, category) if `host` is or sits under a known tracker domain."""
    host = host.lower().rstrip(".")
    for suffix, category in KNOWN_TRACKERS.items():
        if host == suffix or host.endswith("." + suffix):
            return suffix, category
    return None


def find_known_trackers(hosts: Iterable[str]) -> dict[str, tuple[str, str]]:
    """{host: (matched suffix, category)} for every host that matches."""
    found: dict[str, tuple[str, str]] = {}
    for host in hosts:
        match = match_known_tracker(host)
        if match:
            found[host] = match
    return found


def base_domain(host: str) -> str:
    """Heuristic registrable domain. IP literals are returned unchanged."""
    host = host.lower().rstrip(".")
    if _IPV4_RE.match(host) or ":" in host:
        return host
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    if len(labels[-1]) == 2 and labels[-2] in _SECOND_LEVEL_LABELS:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def hosts_by_package(events: Iterable[TraceEvent]) -> dict[str, set[str]]:
    """Contacted hosts per package, from NETWORK_CONNECT events."""
    result: dict[str, set[str]] = {}
    for event in events:
        if event.event_type != EventType.NETWORK_CONNECT:
            continue
        host = event.details.get("host")
        if host:
            result.setdefault(event.package_name, set()).add(str(host))
    return result


def _domains_by_package(host_map: dict[str, set[str]], ignore_domains: set[str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for package, hosts in host_map.items():
        domains = {base_domain(h) for h in hosts} - ignore_domains
        if domains:
            out[package] = domains
    return out


def shared_domains(
    host_map: dict[str, set[str]],
    min_apps: int = 2,
    ignore_domains: set[str] | None = None,
) -> list[SharedDomain]:
    """Domains contacted by at least `min_apps` distinct packages, most
    widely shared first. Loopback/IP-literal destinations are included as
    their own 'domain'; pass them via `ignore_domains` to drop noise."""
    ignore = ignore_domains or set()
    domain_pkgs: dict[str, set[str]] = {}
    domain_hosts: dict[str, set[str]] = {}
    for package, hosts in host_map.items():
        for host in hosts:
            domain = base_domain(host)
            if domain in ignore:
                continue
            domain_pkgs.setdefault(domain, set()).add(package)
            domain_hosts.setdefault(domain, set()).add(host)

    results = []
    for domain, packages in domain_pkgs.items():
        if len(packages) < min_apps:
            continue
        category = None
        for host in domain_hosts[domain] | {domain}:
            match = match_known_tracker(host)
            if match:
                category = match[1]
                break
        results.append(SharedDomain(domain=domain, packages=sorted(packages), known_category=category))

    results.sort(key=lambda s: (-len(s.packages), s.domain))
    return results


def similar_apps(
    host_map: dict[str, set[str]],
    min_jaccard: float = 0.3,
    ignore_domains: set[str] | None = None,
) -> list[SimilarPair]:
    """Package pairs whose contacted base-domain sets overlap by at least
    `min_jaccard` - candidates for sharing an embedded third-party SDK."""
    domains = _domains_by_package(host_map, ignore_domains or set())
    packages = sorted(domains)
    pairs: list[SimilarPair] = []
    for i, a in enumerate(packages):
        for b in packages[i + 1:]:
            shared = domains[a] & domains[b]
            if not shared:
                continue
            jaccard = len(shared) / len(domains[a] | domains[b])
            if jaccard >= min_jaccard:
                pairs.append(SimilarPair(a, b, jaccard, sorted(shared)))
    pairs.sort(key=lambda p: (-p.jaccard, p.package_a, p.package_b))
    return pairs
