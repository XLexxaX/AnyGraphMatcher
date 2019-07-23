import mwparserfromhell

def get_raw_text_and_links_from_markup(raw):
    parsed_wikicode = mwparserfromhell.parse(raw)
    text = parsed_wikicode.strip_code()

    replacement_map = dict()
    for link in parsed_wikicode.filter_wikilinks():
        link_text = str(link.text or link.title)
        link_target = str(link.title)
        replacement_map[str(link_text)] = link_target
        replacement_map[link_target] = link_target

    return text, replacement_map