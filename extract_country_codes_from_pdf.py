"""
PDF Country Code Table Extractor for CBP Agriculture
Extracts country codes from PDF tables and updates cbp_acronyms.json

Usage:
    python extract_country_codes_from_pdf.py <path_to_pdf_file>

Example:
    python extract_country_codes_from_pdf.py "C:/Documents/country_codes.pdf"
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


def extract_country_codes(text):
    """
    Extract country codes from table-like structures
    Handles format: Country Name CODE
    Example:
    - Afghanistan AF
    - United States US
    """
    country_codes = {}

    # Split text into lines
    lines = text.split('\n')

    print(f"📊 Processing {len(lines)} lines...")

    # Pattern: Country/territory name followed by 2-letter ISO code at end of line
    # Matches: "Afghanistan AF" or "United States US"
    pattern_name_code = r'^(.+?)\s+([A-Z]{2})$'

    # Alternative pattern: Code first, then name
    # Matches: "AF Afghanistan" or "US United States"
    pattern_code_name = r'^([A-Z]{2})\s+(.+)$'

    found_count = 0

    for line in lines:
        line = line.strip()

        # Skip empty lines or very short lines
        if len(line) < 4:
            continue

        # Skip header lines
        if 'Country' in line or 'Territory' in line or 'ISO' in line or 'Code' in line:
            continue

        # Try pattern 1: Country name followed by code (most common in your format)
        match = re.match(pattern_name_code, line)
        if match:
            name, code = match.groups()
            name = name.strip()
            code = code.strip()

            # Validate it's a real country name (not just uppercase letters)
            # and not a duplicate code pattern
            if len(name) > 2 and not re.match(r'^[A-Z]{2,}$', name):
                country_codes[code] = name
                found_count += 1
                if found_count <= 5:  # Show first 5 matches as feedback
                    print(f"  ✓ Found: {code} = {name}")
                continue

        # Try pattern 2: Code followed by country name
        match = re.match(pattern_code_name, line)
        if match:
            code, name = match.groups()
            name = name.strip()
            code = code.strip()

            # Validate name looks like a real country name
            if len(name) > 2 and re.match(r'^[A-Z][a-zA-Z\s\-,\.()]+$', name):
                country_codes[code] = name
                found_count += 1
                if found_count <= 5:
                    print(f"  ✓ Found: {code} = {name}")
                continue

    if found_count > 5:
        print(f"  ... and {found_count - 5} more")

    return country_codes


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


def interactive_review(new_codes, existing_acronyms):
    """Interactively review and confirm country codes"""
    print(f"\n🔍 Found {len(new_codes)} potential country codes")
    print("=" * 80)

    confirmed = {}
    skipped = []
    updated = []

    for i, (code, name) in enumerate(new_codes.items(), 1):
        print(f"\n[{i}/{len(new_codes)}] {code}: {name}")

        # Check if already exists
        if code in existing_acronyms:
            print(f"    ⚠️  Already exists: {existing_acronyms[code]}")
            choice = input("    Replace? (y/n/skip all existing) [n]: ").strip().lower()

            if choice == 'skip all existing':
                print("    ⏭️  Skipping all existing codes from now on...")
                # Skip this and all future existing ones
                for remaining_code, remaining_name in list(new_codes.items())[i-1:]:
                    if remaining_code not in existing_acronyms:
                        confirmed[remaining_code] = remaining_name
                break
            elif choice == 'y':
                confirmed[code] = name
                updated.append(code)
                print("    ✅ Will replace")
            else:
                skipped.append(code)
                print("    ⏭️  Skipped")
        else:
            choice = input("    Add this? (y/n/yes to all/edit) [y]: ").strip().lower()

            if choice in ['', 'y', 'yes']:
                confirmed[code] = name
                print("    ✅ Added")
            elif choice == 'yes to all':
                # Add this and all remaining new codes
                confirmed[code] = name
                print("    ✅ Added")
                for remaining_code, remaining_name in list(new_codes.items())[i:]:
                    if remaining_code not in existing_acronyms:
                        confirmed[remaining_code] = remaining_name
                print(f"    ✅ Auto-added {len(list(new_codes.items())[i:])} remaining codes")
                break
            elif choice == 'edit':
                new_name = input(f"    Enter new name for {code}: ").strip()
                if new_name:
                    confirmed[code] = new_name
                    print("    ✅ Added with edited name")
                else:
                    skipped.append(code)
                    print("    ⏭️  Skipped (empty name)")
            else:
                skipped.append(code)
                print("    ⏭️  Skipped")

    print("\n" + "=" * 80)
    print(f"📊 Summary:")
    print(f"  ✅ Confirmed: {len(confirmed)} country codes")
    print(f"  🔄 Updated: {len(updated)} country codes")
    print(f"  ⏭️  Skipped: {len(skipped)} country codes")

    return confirmed


def main():
    print("🌍 PDF Country Code Table Extractor")
    print("=" * 80)

    # Check for PDF file argument
    if len(sys.argv) < 2:
        print("❌ Error: No PDF file specified")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} <path_to_pdf_file>")
        print("\nExample:")
        print(f"  python {sys.argv[0]} \"C:/Documents/country_codes.pdf\"")
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

    # DEBUG: Save extracted text to a file so user can see what was extracted
    debug_file = Path(__file__).parent / 'extracted_text_debug.txt'
    with open(debug_file, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"\n📝 DEBUG: Extracted text saved to {debug_file}")
    print("   Review this file to see the actual text format from the PDF\n")

    # Extract country codes
    print("🔍 Searching for country codes in tables...")
    new_codes = extract_country_codes(text)

    if not new_codes:
        print("❌ No country codes found in PDF")
        print("💡 Tip: Make sure the PDF contains table rows like:")
        print("   US  United States")
        print("   CA  Canada")
        sys.exit(0)

    # Load existing acronyms
    existing_acronyms = load_existing_acronyms()
    print(f"📚 Existing acronyms: {len(existing_acronyms)}")

    # Interactive review
    confirmed_codes = interactive_review(new_codes, existing_acronyms)

    if not confirmed_codes:
        print("\n⚠️  No country codes confirmed. No changes made.")
        sys.exit(0)

    # Merge with existing acronyms
    merged_acronyms = {**existing_acronyms, **confirmed_codes}

    # Save to file
    print(f"\n💾 Saving {len(merged_acronyms)} total entries...")
    save_acronyms(merged_acronyms)

    print("\n✅ Done! Country codes have been added to acronyms database.")
    print(f"   Total entries in database: {len(merged_acronyms)}")
    print(f"   New/updated this session: {len(confirmed_codes)}")

    # Prompt to restart MCP server
    print("\n🔄 Next steps:")
    print("   1. Restart the MCP server to load new codes:")
    print("      docker-compose restart mcp-app")
    print("   2. Test with: /acro <country_code>")


if __name__ == "__main__":
    main()
