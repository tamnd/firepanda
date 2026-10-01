"""`to_html` writes each level of a `MultiIndex` in its own cell, as pandas does."""

import firepanda as fp


def _frame():
    index = fp.MultiIndex.from_tuples([(1, "x"), (1, "y"), (2, "z")], names=["a", "b"])
    return fp.DataFrame({"c": [1.5, None, 2.0]}, index=index)


def _rows(text):
    body = text.split("<tbody>")[1].split("</tbody>")[0]
    return [row.split("</tr>")[0].split() for row in body.split("<tr>")[1:]]


def test_each_level_gets_a_cell_and_a_repeat_one_tall_cell():
    rows = _rows(_frame().to_html())
    assert rows[0] == ["<th", 'rowspan="2"', 'valign="top">1</th>', "<th>x</th>", "<td>1.5</td>"]
    assert rows[1] == ["<th>y</th>", "<td>NaN</td>"]
    assert rows[2] == ["<th>2</th>", "<th>z</th>", "<td>2.0</td>"]


def test_sparsify_off_repeats_every_label():
    rows = _rows(_frame().to_html(sparsify=False))
    assert rows[1] == ["<th>1</th>", "<th>y</th>", "<td>NaN</td>"]


def test_the_level_names_get_a_row_of_their_own():
    head = _frame().to_html().split("<thead>")[1].split("</thead>")[0]
    assert head.count("<tr") == 2
    assert "<th>a</th>\n      <th>b</th>\n      <th></th>" in head
    plain = _frame().to_html(index_names=False).split("<thead>")[1].split("</thead>")[0]
    assert plain.count("<tr") == 1
    assert plain.count("<th></th>") == 2


def test_a_tall_cell_runs_across_the_cut_rows():
    index = fp.MultiIndex.from_tuples([("a", n) for n in range(8)], names=["k", "n"])
    rows = _rows(fp.DataFrame({"v": range(8)}, index=index).to_html(max_rows=4))
    assert rows[0][:3] == ["<th", 'rowspan="5"', 'valign="top">a</th>']
    assert rows[2] == ["<th>...</th>", "<td>...</td>"]
    assert rows[3] == ["<th>6</th>", "<td>6</td>"]


def test_a_run_that_ends_before_the_cut_leaves_the_dots_a_cell():
    tuples = [("a", 0), ("a", 1), ("b", 2), ("b", 3), ("c", 4), ("c", 5)]
    index = fp.MultiIndex.from_tuples(tuples)
    rows = _rows(fp.DataFrame({"v": range(6)}, index=index).to_html(max_rows=4))
    assert rows[2] == ["<th>...</th>", "<th>...</th>", "<td>...</td>"]


def test_sparsify_off_cut_rows_get_a_row_of_dots():
    index = fp.MultiIndex.from_tuples([("a", n) for n in range(8)])
    rows = _rows(fp.DataFrame({"v": range(8)}, index=index).to_html(max_rows=4, sparsify=False))
    assert rows[2] == ["<th>...</th>", "<th>...</th>", "<td>...</td>"]
    assert rows[3] == ["<th>a</th>", "<th>6</th>", "<td>6</td>"]


def test_without_the_index_no_label_cells():
    rows = _rows(_frame().to_html(index=False))
    assert rows[0] == ["<td>1.5</td>"]
