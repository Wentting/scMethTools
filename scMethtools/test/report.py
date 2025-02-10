def parse_map_qc_html(html_doc: str) -> Dict[str, Union[float, int]]:
    qc: Dict[str, Union[float, int]] = {}
    soup = BeautifulSoup(html_doc, "html.parser")

    map_stats_table = find_table(soup, name="Mapping Stats (Reads)")
    if map_stats_table is None:
        raise ValueError("Could not find mapping stats table")
    for row in map_stats_table:
        qc_key = qc_key_from_tag(row.td)
        qc_value: Union[int, float] = int_from_tag(row.contents[3])
        qc_pct_key = "pct_" + qc_key
        qc_pct_value = percent_from_tag(row.contents[5])
        qc[qc_key] = qc_value
        qc[qc_pct_key] = qc_pct_value

    fragment_uniqueness_table = find_table(soup, name="Uniqueness (Fragments)")
    if fragment_uniqueness_table is None:
        raise ValueError("Could not find fragment uniqueness table")
    for row in fragment_uniqueness_table:
        qc_key = qc_key_from_tag(row.td)
        if qc_key == "average_unique":
            qc_key = "pct_unique_fragments"
            qc_value = percent_from_tag(row.contents[3])
        else:
            qc_value = int_from_tag(row.contents[3])
        qc[qc_key] = qc_value

    bisulfite_conversion_rate_table = find_table(soup, name="Bisulfite Conversion Rate")
    if bisulfite_conversion_rate_table is None:
        raise ValueError("Could not determine bisulfite conversion rate")
    for row in bisulfite_conversion_rate_table:
        qc_key = qc_key_from_tag(row.td)
        try:
            qc_value = float(string_from_tag(row.contents[3]))
        except ValueError:
            logging.warning(
                f"Could not parse qc value for {qc_key} into float", exc_info=True
            )
            continue
        qc[qc_key] = qc_value

    correct_pairs_table = find_table(soup, name="Correct Pairs")
    if correct_pairs_table is None:
        logging.info("Correct pairs table not found, assuming data is single ended")
        return qc
    for row in correct_pairs_table:
        qc_key = qc_key_from_tag(row.td)
        qc_value = int_from_tag(row.contents[3])
        qc[qc_key] = qc_value

    return qc