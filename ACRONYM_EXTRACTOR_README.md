# PDF Acronym Extractor for CBP Agriculture

Automatically extract acronyms from PDF documents and add them to your CBP acronym database.

## Quick Start

```bash
python extract_acronyms_from_pdf.py "path/to/your/document.pdf"
```

## Features

✅ **Automatic Detection** - Finds acronyms using multiple patterns:
- `ABC (Full Name)` - Most common format
- `Full Name (ABC)` - Reverse format
- `ABC: Full Name` - Colon separated
- `ABC - Full Name` - Dash separated

✅ **Interactive Review** - Review each acronym before adding
- Accept, skip, or edit definitions
- Bulk operations (yes to all, skip all existing)
- See which acronyms already exist

✅ **Smart Merging** - Safely updates existing database
- Preserves existing acronyms
- Prompts before overwriting
- Alphabetically sorted output

✅ **Auto-install** - Automatically installs PyPDF2 if needed

## Usage Examples

### Basic Usage
```bash
python extract_acronyms_from_pdf.py "C:/Documents/CBP_Manual.pdf"
```

### With Quotes (for paths with spaces)
```bash
python extract_acronyms_from_pdf.py "C:/My Documents/CBP Agriculture Guide.pdf"
```

### From Current Directory
```bash
python extract_acronyms_from_pdf.py "./manual.pdf"
```

## Interactive Session Example

```
🔤 CBP Agriculture PDF Acronym Extractor
================================================================================
📄 Reading PDF: C:/Documents/manual.pdf
📖 Total pages: 45
✅ Extracted 125000 characters from PDF

🔍 Searching for acronyms...
📚 Existing acronyms: 34

🔍 Found 23 potential acronyms
================================================================================

[1/23] PACA: Perishable Agricultural Commodities Act
    Add this? (y/n/yes to all/edit) [y]: y
    ✅ Added

[2/23] AGC: Agriculture
    ⚠️  Already exists: Agriculture
    Replace? (y/n/skip all existing) [n]: n
    ⏭️  Skipped

[3/23] AQIM: Agriculture Quarantine Inspection Monitoring
    Add this? (y/n/yes to all/edit) [y]: edit
    Enter new definition for AQIM: Agricultural Quarantine Inspection Monitoring
    ✅ Added with edited definition

[4/23] CBPO: Customs and Border Protection Officer
    Add this? (y/n/yes to all/edit) [y]: yes to all
    ✅ Added
    ✅ Auto-added 19 remaining acronyms
```

## Options During Review

| Input | Action |
|-------|--------|
| `y` or `Enter` | Accept and add acronym |
| `n` | Skip this acronym |
| `yes to all` | Add this and all remaining NEW acronyms |
| `skip all existing` | Skip all acronyms that already exist |
| `edit` | Edit the definition before adding |

## What Happens Next

After extraction completes:

1. **JSON file updated** - `cbp_acronyms.json` is saved with new acronyms
2. **Restart MCP server** - Run: `docker-compose restart mcp-app`
3. **Test it** - Use `/acro <acronym>` in the chat interface

## Requirements

- Python 3.7+
- PyPDF2 (auto-installed if missing)

## Troubleshooting

### No acronyms found
- Make sure the PDF contains searchable text (not scanned images)
- Check that acronyms follow standard formats: `ABC (Full Name)`
- Try extracting text manually to verify PDF is readable

### PyPDF2 installation fails
```bash
pip install PyPDF2
```

### File not found error
- Use absolute paths with quotes
- Check file extension is `.pdf`
- Verify file exists at specified location

## Tips for Best Results

✅ **DO:**
- Use PDFs with searchable text
- Review each acronym carefully
- Use "edit" to fix incorrect definitions
- Use "yes to all" for trusted documents

❌ **DON'T:**
- Skip reviewing acronyms (may add incorrect ones)
- Use scanned PDFs without OCR
- Forget to restart MCP server after updating

## File Location

The script updates: `cbp_acronyms.json` in the same directory

## Support

If you encounter issues:
1. Check the PDF is text-based (not scanned)
2. Verify Python 3.7+ is installed
3. Try running with a simple test PDF first
