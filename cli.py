"""
Command Line Interface (CLI) for College Information FAQ Chatbot
CodeAlpha Artificial Intelligence Internship - Task 2
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chatbot import CollegeFAQChatbot


# ANSI escape codes for clean terminal styling
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_banner(threshold: float, dataset_len: int) -> None:
    print(f"{CYAN}{BOLD}======================================================================{RESET}")
    print(f"{CYAN}{BOLD}       ApexTech College Information FAQ Chatbot (CodeAlpha Task 2)    {RESET}")
    print(f"{CYAN}{BOLD}======================================================================{RESET}")
    print(f" Loaded {dataset_len} College FAQs. Cosine Similarity Cutoff: {BOLD}{threshold:.2f}{RESET}")
    print(f" Commands: {YELLOW}'exit'{RESET} or {YELLOW}'quit'{RESET} to close | {YELLOW}'debug'{RESET} to toggle debug mode | {YELLOW}'categories'{RESET} to list topics")
    print(f"{CYAN}----------------------------------------------------------------------{RESET}\n")


def run_cli():
    parser = argparse.ArgumentParser(
        description="Run the College Information FAQ Chatbot in interactive CLI mode."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/faqs.csv",
        help="Path to FAQ dataset (CSV or JSON). Default: data/faqs.csv",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Similarity confidence threshold (0.0 to 1.0). Default: 0.35",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable developer debug mode displaying similarity scores and matched features.",
    )
    args = parser.parse_args()

    try:
        bot = CollegeFAQChatbot(
            dataset_path=args.dataset,
            similarity_threshold=args.threshold,
            top_k=3,
            include_keywords=True,
        )
    except Exception as exc:
        print(f"{RED}{BOLD}Failed to initialize College FAQ Chatbot: {exc}{RESET}")
        sys.exit(1)

    debug_mode = args.debug
    print_banner(bot.similarity_threshold, len(bot.records))

    while True:
        try:
            user_input = input(f"{BOLD}Student: {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{YELLOW}Goodbye! Exiting College FAQ Chatbot.{RESET}")
            break

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in {"exit", "quit", "q", "bye"}:
            print(f"\n{YELLOW}Thank you for consulting ApexTech College FAQ Assistant. Have a great day!{RESET}")
            break

        if lowered == "debug":
            debug_mode = not debug_mode
            status = f"{GREEN}ENABLED{RESET}" if debug_mode else f"{RED}DISABLED{RESET}"
            print(f"{YELLOW}[System] Debug mode is now {status}.{RESET}\n")
            continue

        if lowered == "categories":
            cats = bot.get_categories()
            print(f"\n{CYAN}{BOLD}Available College FAQ Categories:{RESET}")
            for c in cats:
                count = len(bot.get_faqs_by_category(c))
                print(f" - {BOLD}{c}{RESET} ({count} FAQs)")
            print()
            continue

        # Process question
        result = bot.ask(user_input, return_debug=debug_mode)
        score = result["confidence_score"]

        if result["is_fallback"]:
            print(f"\n{RED}{BOLD}Bot:{RESET} {result['answer']}")
        else:
            print(f"\n{GREEN}{BOLD}Bot:{RESET} {result['answer']}")

        # Show match suggestions if available
        if result["suggestions"] and len(result["suggestions"]) > 1:
            print(f"\n{DIM}Related FAQs you might find helpful:{RESET}")
            for sug in result["suggestions"][:2]:
                if sug["question"] != result["matched_question"]:
                    print(f"  {DIM}* {sug['question']} (score: {sug['score']:.2f}){RESET}")

        # Developer Debug mode output
        if debug_mode and result.get("debug"):
            dbg = result["debug"]
            print(f"\n{YELLOW}--- [DEBUG MODE] ---{RESET}")
            print(f" Cleaned Query: {dbg.get('cleaned_query')}")
            print(f" Best Match FAQ ID: {result['matched_faq_id']}")
            print(f" Best Stored Question: {result['matched_question']}")
            print(f" Category: {result['category']}")
            print(f" Cosine Similarity Score: {score:.4f} (Threshold: {bot.similarity_threshold:.2f})")
            if "explainability" in dbg and dbg["explainability"]:
                exp = dbg["explainability"]
                terms = [f"{t['term']} ({t['contribution']:.3f})" for t in exp.get("overlapping_features", [])[:5]]
                print(f" Overlapping TF-IDF Features: {', '.join(terms) if terms else 'None'}")
            print(f"{YELLOW}--------------------{RESET}")

        print()


if __name__ == "__main__":
    run_cli()
