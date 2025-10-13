"""
PDF Acronym Extractor for CBP Agriculture
Extracts acronyms from PDF documents and updates cbp_acronyms.json

Usage:
    python extract_acronyms_from_pdf.py <path_to_pdf_file>

Example:
    python extract_acronyms_from_pdf.py "C:/Documents/cbp_manual.pdf"
"""

import sys
import json
import re
from pathlib import Path

try:
    import PyPDF2
except ImportError:
    print("❌ PyPDF2 not installed. Installing now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
    import PyPDF2


def extract_text_from_pdf(pdf_path):
    """Extract all text from PDF file"""
    print(f"📄 Reading PDF: {pdf_path}")

    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""

            print(f"📖 Total pages: {len(pdf_reader.pages)}")

            for i, page in enumerate(pdf_reader.pages, 1):
                print(f"  Reading page {i}/{len(pdf_reader.pages)}...", end='\r')
                text += page.extract_text() + "\n"

            print(f"\n✅ Extracted {len(text)} characters from PDF")
            return text

    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return None


def find_acronyms(text):
    """
    Find acronyms in text using multiple patterns:
    1. ABC (Full Name Here)
    2. Full Name Here (ABC)
    3. ABC: Full Name Here
    4. ABC - Full Name Here
    """
    acronyms = {}

    # Pattern 1: ABC (Full Definition) - most common
    # Matches: AGC (Agriculture), HSUSA (Harmonized System – United States of America)
    pattern1 = r'\b([A-Z]{2,})\s*\(([^)]+)\)'
    matches1 = re.findall(pattern1, text)
    for acronym, definition in matches1:
        # Clean up definition
        definition = definition.strip()
        # Skip if definition looks like it's just more acronyms or numbers
        if not re.match(r'^[A-Z0-9\s,]+$', definition) and len(definition) > 3:
            acronyms[acronym] = definition

    # Pattern 2: Full Definition (ABC)
    # Matches: Agriculture (AGC)
    pattern2 = r'([A-Z][a-zA-Z\s\-–]+?)\s*\(([A-Z]{2,})\)'
    matches2 = re.findall(pattern2, text)
    for definition, acronym in matches2:
        definition = definition.strip()
        if len(definition) > 3 and acronym not in acronyms:
            acronyms[acronym] = definition

    # Pattern 3: ABC: Full Definition
    # Matches: AGC: Agriculture
    pattern3 = r'\b([A-Z]{2,})\s*:\s*([A-Z][a-zA-Z\s\-–]+?)(?:\.|;|\n|$)'
    matches3 = re.findall(pattern3, text)
    for acronym, definition in matches3:
        definition = definition.strip()
        if len(definition) > 3 and acronym not in acronyms:
            acronyms[acronym] = definition

    # Pattern 4: ABC - Full Definition
    # Matches: AGC - Agriculture
    pattern4 = r'\b([A-Z]{2,})\s*[-–]\s*([A-Z][a-zA-Z\s\-–]+?)(?:\.|;|\n|$)'
    matches4 = re.findall(pattern4, text)
    for acronym, definition in matches4:
        definition = definition.strip()
        if len(definition) > 3 and acronym not in acronyms:
            acronyms[acronym] = definition

    return acronyms


def load_existing_acronyms():
    """Load existing acronyms from JSON file"""
    json_path = Path(__file__).parent / 'cbp_acronyms.json'

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  {json_path} not found. Will create new file.")
        return {}
    except json.JSONDecodeError as e:
        print(f"⚠️  Error parsing JSON: {e}")
        return {}


def save_acronyms(acronyms):
    """Save acronyms to JSON file"""
    json_path = Path(__file__).parent / 'cbp_acronyms.json'

    # Sort acronyms alphabetically
    sorted_acronyms = dict(sorted(acronyms.items()))

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_acronyms, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved to {json_path}")


def interactive_review(new_acronyms, existing_acronyms):
    """Interactively review and confirm acronyms"""
    print(f"\n🔍 Found {len(new_acronyms)} potential acronyms")
    print("=" * 80)

    confirmed = {}
    skipped = []
    updated = []

    for i, (acronym, definition) in enumerate(new_acronyms.items(), 1):
        print(f"\n[{i}/{len(new_acronyms)}] {acronym}: {definition}")

        # Check if already exists
        if acronym in existing_acronyms:
            print(f"    ⚠️  Already exists: {existing_acronyms[acronym]}")
            choice = input("    Replace? (y/n/skip all existing) [n]: ").strip().lower()

            if choice == 'skip all existing':
                print("    ⏭️  Skipping all existing acronyms from now on...")
                # Skip this and all future existing ones
                for remaining_acronym, remaining_def in list(new_acronyms.items())[i-1:]:
                    if remaining_acronym not in existing_acronyms:
                        confirmed[remaining_acronym] = remaining_def
                break
            elif choice == 'y':
                confirmed[acronym] = definition
                updated.append(acronym)
                print("    ✅ Will replace")
            else:
                skipped.append(acronym)
                print("    ⏭️  Skipped")
        else:
            choice = input("    Add this? (y/n/yes to all/edit) [y]: ").strip().lower()

            if choice in ['', 'y', 'yes']:
                confirmed[acronym] = definition
                print("    ✅ Added")
            elif choice == 'yes to all':
                # Add this and all remaining new acronyms
                confirmed[acronym] = definition
                print("    ✅ Added")
                for remaining_acronym, remaining_def in list(new_acronyms.items())[i:]:
                    if remaining_acronym not in existing_acronyms:
                        confirmed[remaining_acronym] = remaining_def
                print(f"    ✅ Auto-added {len(list(new_acronyms.items())[i:])} remaining acronyms")
                break
            elif choice == 'edit':
                new_def = input(f"    Enter new definition for {acronym}: ").strip()
                if new_def:
                    confirmed[acronym] = new_def
                    print("    ✅ Added with edited definition")
                else:
                    skipped.append(acronym)
                    print("    ⏭️  Skipped (empty definition)")
            else:
                skipped.append(acronym)
                print("    ⏭️  Skipped")

    print("\n" + "=" * 80)
    print(f"📊 Summary:")
    print(f"  ✅ Confirmed: {len(confirmed)} acronyms")
    print(f"  🔄 Updated: {len(updated)} acronyms")
    print(f"  ⏭️  Skipped: {len(skipped)} acronyms")

    return confirmed


def main():
    print("🔤 PDF Acronym Extractor")
    print("=" * 80)

    # Check for PDF file argument
    if len(sys.argv) < 2:
        print("❌ Error: No PDF file specified")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} <path_to_pdf_file>")
        print("\nExample:")
        print(f"  python {sys.argv[0]} \"C:/Documents/cbp_manual.pdf\"")
        sys.exit(1)

    pdf_path = sys.argv[1]

    # Check if file exists
    if not Path(pdf_path).exists():
        print(f"❌ Error: File not found: {pdf_path}")
        sys.exit(1)

    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("❌ Failed to extract text from PDF")
        sys.exit(1)

    # Find acronyms
    print("\n🔍 Searching for acronyms...")
    new_acronyms = find_acronyms(text)

    if not new_acronyms:
        print("❌ No acronyms found in PDF")
        print("💡 Tip: Make sure the PDF contains text like 'ABC (Full Name)' or 'Full Name (ABC)'")
        sys.exit(0)

    # Load existing acronyms
    existing_acronyms = load_existing_acronyms()
    print(f"📚 Existing acronyms: {len(existing_acronyms)}")

    # Interactive review
    confirmed_acronyms = interactive_review(new_acronyms, existing_acronyms)

    if not confirmed_acronyms:
        print("\n⚠️  No acronyms confirmed. No changes made.")
        sys.exit(0)

    # Merge with existing acronyms
    merged_acronyms = {**existing_acronyms, **confirmed_acronyms}

    # Save to file
    print(f"\n💾 Saving {len(merged_acronyms)} total acronyms...")
    save_acronyms(merged_acronyms)

    print("\n✅ Done! Acronyms have been updated.")
    print(f"   Total acronyms in database: {len(merged_acronyms)}")
    print(f"   New/updated this session: {len(confirmed_acronyms)}")

    # Prompt to restart MCP server
    print("\n🔄 Next steps:")
    print("   1. Restart the MCP server to load new acronyms:")
    print("      docker-compose restart mcp-app")
    print("   2. Test with: /acro <acronym>")


if __name__ == "__main__":
    main()
