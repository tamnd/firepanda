"""A query text in, rows out.

Every other test in the SQL front end checks one stage against the stage's own
idea of an answer. The transform tests check SQL printed back, the lowering
tests check the text `explain` produces, and the pipeline tests check frames
built by hand. All of them can pass while the stages do not fit together, which
is what this file is for: it names no stage and asserts nothing about a plan, it
writes a query and checks the rows.

The queries are deliberately small and the numbers are deliberately not in
order, so that a filter that keeps a middle range is not a slice and a sort that
does nothing is visible. The frames have three chunks for the same reason they
do in the pipeline tests, which is that a chunk boundary is where an off by one
in a position lives.

The refusals matter as much as the answers. A shape nothing runs yet has to say
so by name, because a query that quietly returned the wrong rows would be the
kind of defect that only a differential run against DuckDB finds, and a refusal
is something a caller can act on.

Part 2 of 3. The fixtures are in tests/support/sql_run.mojo.
"""


from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.chunked import ChunkedArray
from firepanda.array.strings import StringBuilder
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.dtype.temporal import TimeUnit
from firepanda.frame.frame import DataFrame
from firepanda.sql.catalog import Catalog
from firepanda.sql.run import run

from tests.support.sql_run import (
    _marks,
    answer,
    cuts,
    days,
    dupes,
    gapped,
    gappy,
    gaps,
    glyphs,
    hits,
    lengths,
    moments,
    numbers,
    padded,
    read_back,
    sales,
    same,
    session,
    shifts,
    shops,
    stock,
    tiers,
    truths,
    visits,
    words,
)


def test_an_is_null_keeps_the_rows_with_nothing_in_them() raises:
    same(
        gapped(
            run("SELECT mark FROM gappy WHERE mark IS NULL", session()), "mark"
        ),
        [-1, -1],
        "mark",
    )


def test_an_is_not_null_keeps_the_others() raises:
    same(
        answer("SELECT mark FROM gappy WHERE mark IS NOT NULL", "mark"),
        [4, 4, 9, 1],
        "mark",
    )


def test_an_is_null_in_a_select_list_answers_yes_or_no_for_every_row() raises:
    # Never a null itself, whatever the column under it holds, which is what
    # tells this apart from `mark = NULL` and is the reason SQL has the words.
    same(
        truths(
            run("SELECT mark IS NULL AS gone FROM gappy", session()), "gone"
        ),
        [0, 0, 1, 0, 1, 0],
        "gone",
    )


def test_the_one_word_spellings_mean_the_same_two_tests() raises:
    # `ISNULL` and `NOTNULL` are postfix words rather than functions, and the
    # parser folds each of them into the node the two word form builds.
    same(
        answer("SELECT mark FROM gappy WHERE mark NOTNULL", "mark"),
        [4, 4, 9, 1],
        "mark",
    )
    same(
        gapped(
            run("SELECT mark FROM gappy WHERE mark ISNULL", session()), "mark"
        ),
        [-1, -1],
        "mark",
    )


def test_an_is_null_reads_a_column_of_text_too() raises:
    same(answer("SELECT n FROM words WHERE word IS NULL", "n"), [6], "n")


def test_an_is_true_keeps_the_rows_the_comparison_held_for() raises:
    # `mark > 3` is true, true, null, true, null, false down the six rows, so
    # each of the four tests below keeps a different set and no two of them
    # would agree if the null were being read as a false.
    same(
        answer("SELECT mark FROM gappy WHERE (mark > 3) IS TRUE", "mark"),
        [4, 4, 9],
        "mark",
    )


def test_an_is_false_keeps_the_row_it_did_not_hold_for() raises:
    same(
        answer("SELECT mark FROM gappy WHERE (mark > 3) IS FALSE", "mark"),
        [1],
        "mark",
    )


def test_an_is_not_true_keeps_the_nulls_with_the_false() raises:
    same(
        gapped(
            run(
                "SELECT mark FROM gappy WHERE (mark > 3) IS NOT TRUE", session()
            ),
            "mark",
        ),
        [-1, -1, 1],
        "mark",
    )


def test_an_is_not_false_keeps_the_nulls_with_the_true() raises:
    same(
        gapped(
            run(
                "SELECT mark FROM gappy WHERE (mark > 3) IS NOT FALSE",
                session(),
            ),
            "mark",
        ),
        [4, 4, -1, 9, -1],
        "mark",
    )


def test_a_yes_written_out_keeps_every_row_and_a_no_keeps_none() raises:
    # The word is written in capitals by the parser whatever the query spelled
    # it with, and reading it as lower case made every one of them a no.
    same(
        gapped(run("SELECT mark FROM gappy WHERE true", session()), "mark"),
        [4, 4, -1, 9, -1, 1],
        "mark",
    )
    assert_equal(
        len(run("SELECT mark FROM gappy WHERE FALSE", session())),
        0,
        "a no keeps nothing",
    )


def test_a_case_change_rewrites_every_row() raises:
    var up = cuts("SELECT upper(word) AS piece FROM words")
    assert_equal(len(up), 7, "one answer per row")
    assert_equal(up[0], "APPLE", "a row was raised")
    assert_equal(up[4], "", "an empty row stays empty")
    assert_equal(up[5], "null", "and a null stays a null")
    var down = cuts("SELECT lower(upper(word)) AS piece FROM words")
    assert_equal(down[0], "apple", "and lowering it again gives it back")


def test_a_case_change_reads_characters_and_not_bytes() raises:
    # `glyphs` holds a row that is not ASCII and a row that has no case at all,
    # which are the two a case change written over the payload gets wrong.
    var up = cuts("SELECT upper(word) AS piece FROM glyphs")
    assert_equal(up[0], "ABC", "the ASCII row is the easy one")
    assert_equal(up[1], "HÉLLO", "an accented letter raises to its own capital")
    assert_equal(up[2], "日本語です", "and a script with no case is left alone")
    var down = cuts("SELECT lower(word) AS piece FROM glyphs WHERE n = 2")
    assert_equal(down[0], "héllo", "and it lowers back to what it was")


def test_the_other_two_names_for_a_case_change_answer_the_same() raises:
    var up = cuts("SELECT ucase(word) AS piece FROM words WHERE n = 1")
    var down = cuts("SELECT lcase(word) AS piece FROM words WHERE n = 1")
    assert_equal(up[0], "APPLE", "ucase is upper")
    assert_equal(down[0], "apple", "and lcase is lower")


def test_a_case_change_is_a_column_like_any_other() raises:
    # The answer goes under a `WHERE` and through another function without
    # either of them knowing what made it, which is the thing a new node has
    # to earn rather than be given.
    same(
        answer("SELECT n FROM words WHERE upper(word) = 'BANANA'", "n"),
        [3],
        "n",
    )
    same(
        answer("SELECT length(lower(word)) AS c FROM glyphs WHERE n = 3", "c"),
        [5],
        "c",
    )


def test_a_regexp_matches_looks_anywhere_in_the_row() raises:
    # The name says matches and the answer is found somewhere in the row, which
    # is DuckDB's reading. Written as a Python `re.match` this would keep only
    # the rows that start with the pattern.
    same(
        answer("SELECT n FROM words WHERE regexp_matches(word, 'an')", "n"),
        [3],
        "n",
    )
    same(
        answer("SELECT n FROM words WHERE regexp_matches(word, 'ap')", "n"),
        [1, 2, 4, 7],
        "n",
    )


def test_a_regexp_matches_leaves_the_anchors_to_the_caller() raises:
    same(
        answer("SELECT n FROM words WHERE regexp_matches(word, '^ap')", "n"),
        [1, 2],
        "n",
    )
    same(
        answer("SELECT n FROM words WHERE regexp_matches(word, '^ap$')", "n"),
        [],
        "n",
    )


def test_a_regexp_matches_reads_the_syntax_a_like_has_no_way_to_write() raises:
    # A count, a class and an alternation, none of which a `LIKE` can say, and
    # all three of which are the reason this function is worth having.
    same(
        answer("SELECT n FROM words WHERE regexp_matches(word, 'p{2}')", "n"),
        [1, 7],
        "n",
    )
    same(
        answer(
            "SELECT n FROM words WHERE regexp_matches(word, '^(gr|ba)')", "n"
        ),
        [3, 4],
        "n",
    )


def test_a_regexp_matches_answers_every_row_and_a_null_for_the_null() raises:
    same(
        truths(
            run(
                "SELECT regexp_matches(word, 'e$') AS hit FROM words",
                session(),
            ),
            "hit",
        ),
        [1, 0, 0, 1, 0, -1, 1],
        "hit",
    )


def test_a_regexp_replace_swaps_the_first_match_and_stops() raises:
    # DuckDB replaces one match without `g` and this is the only test in the
    # file that can tell the difference, every other pattern here matching at
    # most once anyway.
    var got = cuts("SELECT regexp_replace(word, 'p', 'P') AS piece FROM words")
    assert_equal(len(got), 7, "one answer per row")
    assert_equal(got[0], "aPple", "the first p went and the second stayed")
    assert_equal(got[6], "Pineapple", "and the same on a row with three")


def test_a_regexp_replace_with_g_swaps_every_match() raises:
    var got = cuts(
        "SELECT regexp_replace(word, 'p', 'P', 'g') AS piece FROM words"
    )
    assert_equal(got[0], "aPPle", "both of them this time")
    assert_equal(got[6], "PineaPPle", "and all three of them")


def test_a_regexp_replace_writes_the_groups_the_replacement_names() raises:
    # The shape ClickBench q28 is written in, which is the query this went in
    # for: pull the host out of a URL and keep nothing else.
    var got = cuts(
        "SELECT regexp_replace('http://www.example.com/a/b',"
        " '^https?://(?:www\\.)?([^/]+)/.*$', '\\1') AS piece FROM words"
        " WHERE n = 1"
    )
    assert_equal(got[0], "example.com", "the group is what came out")
    var two = cuts(
        "SELECT regexp_replace(word, '^(.)(.)', '\\2\\1') AS piece FROM words"
        " WHERE n = 3"
    )
    assert_equal(two[0], "abnana", "and two groups come out in the order asked")


def test_a_regexp_replace_leaves_a_row_with_no_match_as_it_was() raises:
    var got = cuts("SELECT regexp_replace(word, 'zz', '!') AS piece FROM words")
    assert_equal(got[0], "apple", "a row with no match is handed back")
    assert_equal(got[4], "", "an empty row stays empty")
    assert_equal(got[5], "null", "and a null stays a null")


def test_a_regexp_replace_is_a_column_like_any_other() raises:
    same(
        answer(
            (
                "SELECT n FROM words WHERE regexp_replace(word, 'a', 'o') ="
                " 'opple'"
            ),
            "n",
        ),
        [1],
        "n",
    )


def test_a_regular_expression_against_a_column_is_refused() raises:
    with assert_raises(contains="have to be written out"):
        _ = run(
            "SELECT n FROM words WHERE regexp_matches(word, word)", session()
        )
    with assert_raises(contains="have to be written out"):
        _ = run(
            "SELECT regexp_replace(word, 'a', word) AS piece FROM words",
            session(),
        )


def test_a_pattern_this_library_cannot_read_is_refused_by_name() raises:
    with assert_raises(contains="cannot run the pattern"):
        _ = run(
            "SELECT n FROM words WHERE regexp_matches(word, '(')", session()
        )
    # A backreference is Python's syntax and not RE2's, and SQL is RE2 here, so
    # this one is refused for being wrong rather than for being missing.
    with assert_raises(contains="RE2 has no backreference"):
        _ = run(
            "SELECT n FROM words WHERE regexp_matches(word, '(a)\\1')",
            session(),
        )


def test_a_replacement_the_pattern_cannot_fill_is_refused() raises:
    with assert_raises(contains="cannot read the replacement"):
        _ = run(
            "SELECT regexp_replace(word, 'a', '\\1') AS piece FROM words",
            session(),
        )


def test_an_option_other_than_g_is_refused_rather_than_dropped() raises:
    # Reading `i` and answering a case sensitive match would be wrong with
    # nothing anywhere to say so, which is the whole reason for the refusal.
    with assert_raises(contains="the only one answered is 'g'"):
        _ = run(
            "SELECT regexp_replace(word, 'a', 'o', 'i') AS piece FROM words",
            session(),
        )


def test_a_regular_expression_over_a_number_says_so() raises:
    with assert_raises(contains="'regexp_matches' reads text"):
        _ = run("SELECT n FROM words WHERE regexp_matches(n, 'a')", session())
    with assert_raises(contains="'regexp_replace' takes three arguments"):
        _ = run(
            "SELECT regexp_replace(word, 'a') AS piece FROM words",
            session(),
        )


def test_a_trim_takes_the_spaces_off_both_ends() raises:
    var got = cuts("SELECT trim(word) AS piece FROM padded")
    assert_equal(len(got), 6, "one answer per row")
    assert_equal(got[0], "hi", "both ends came off")
    assert_equal(got[2], "xxaxx", "a row with nothing on its ends is as it was")
    assert_equal(got[3], "", "an empty row stays empty")
    assert_equal(got[4], "", "and a row that is nothing but spaces becomes one")
    assert_equal(got[5], "null", "and a null stays a null")


def test_a_trim_leaves_a_tab_where_duckdb_leaves_it() raises:
    # A tab is whitespace to Python and is not one of the Zs characters, so
    # `TRIM` hands this row back exactly as it arrived. `.str.strip()` on the
    # same column would not, and that difference is deliberate.
    var got = cuts("SELECT trim(word) AS piece FROM padded")
    assert_equal(got[1], "\tgo\t", "the tabs stayed on")


def test_the_one_sided_trims_work_on_the_end_they_name() raises:
    var left = cuts("SELECT ltrim(word) AS piece FROM padded WHERE n = 1")
    var right = cuts("SELECT rtrim(word) AS piece FROM padded WHERE n = 1")
    assert_equal(left[0], "hi  ", "the far end was left alone")
    assert_equal(right[0], "  hi", "and the near end was")


def test_a_trim_of_a_set_takes_any_of_those_characters_off() raises:
    var got = cuts("SELECT trim(word, 'x') AS piece FROM padded WHERE n = 3")
    assert_equal(got[0], "a", "the characters in the set came off")


def test_a_trim_set_is_a_set_and_not_a_prefix() raises:
    # `trim('abcxcba', 'abc')` is `x` in DuckDB, which is the reading this
    # matches. `apricot` loses the a and the p and stops at the r.
    var got = cuts("SELECT trim(word, 'ap') AS piece FROM words WHERE n = 2")
    assert_equal(got[0], "ricot", "every leading character in the set came off")


def test_a_trim_of_a_set_leaves_the_whitespace_alone() raises:
    var got = cuts("SELECT trim(word, 'x') AS piece FROM padded WHERE n = 1")
    assert_equal(got[0], "  hi  ", "a set does not mean whitespace as well")


def test_the_keyword_spelling_of_a_trim_answers_the_same() raises:
    var written = cuts(
        "SELECT TRIM(BOTH ' ' FROM word) AS piece FROM padded WHERE n = 1"
    )
    var called = cuts("SELECT trim(word, ' ') AS piece FROM padded WHERE n = 1")
    assert_equal(written[0], "hi", "the keyword spelling ran")
    assert_equal(called[0], "hi", "and the call spelling agrees")


def test_a_trim_whose_set_is_a_column_is_refused_while_it_lowers() raises:
    # There is no kernel that reads a new set for every row, so this is refused
    # rather than answered with the first row's set for all of them.
    with assert_raises(contains="have to be written out"):
        _ = run("SELECT trim(word, word) AS piece FROM padded", session())


def test_a_search_counts_from_one_and_says_zero_for_a_miss() raises:
    # `words` is apple, apricot, banana, grape, the empty string, a null and
    # pineapple, and the null is left out because a missing answer is the next
    # test rather than this one.
    var got = answer(
        "SELECT strpos(word, 'an') AS c FROM words WHERE n <> 6", "c"
    )
    same(got, [0, 0, 2, 0, 0, 0], "c")


def test_a_search_that_hits_twice_answers_the_first_one() raises:
    var got = answer(
        "SELECT strpos(word, 'na') AS c FROM words WHERE n = 3", "c"
    )
    same(got, [3], "c")


def test_a_search_of_a_row_with_nothing_in_it_has_no_answer() raises:
    # DuckDB answers null rather than zero, and the two are different things to
    # anything that folds the column afterwards.
    var out = run(
        "SELECT strpos(word, 'a') AS c FROM words WHERE n = 6", session()
    )
    var col = out.column("c").as_typed[DType.int64]()
    assert_equal(len(col), 1, "one row")
    assert_true(not col.is_valid(0), "and nothing in it")


def test_a_search_counts_characters_and_not_bytes() raises:
    # `glyphs` row three is five characters and fifteen bytes, so a search that
    # counted bytes would answer seven here.
    var got = answer(
        "SELECT strpos(word, '語') AS c FROM glyphs WHERE n = 3", "c"
    )
    same(got, [3], "c")


def test_a_search_for_nothing_answers_the_first_character() raises:
    var got = answer("SELECT strpos(word, '') AS c FROM words WHERE n = 1", "c")
    same(got, [1], "c")


def test_the_keyword_spelling_of_a_search_answers_the_same() raises:
    var written = answer(
        "SELECT POSITION('an' IN word) AS c FROM words WHERE n = 3", "c"
    )
    var called = answer(
        "SELECT strpos(word, 'an') AS c FROM words WHERE n = 3", "c"
    )
    same(written, [2], "the keyword spelling ran")
    same(called, [2], "and the call spelling agrees")


def test_a_search_for_a_column_is_refused_while_it_lowers() raises:
    # There is no kernel that reads a new needle for every row, so this is
    # refused rather than answered with the first row's needle for all of them.
    with assert_raises(contains="have to be written out"):
        _ = run("SELECT strpos(word, word) AS c FROM words", session())


def test_a_search_folds_the_way_clickbench_folds_one() raises:
    # The shape ClickBench uses it in: a search worked out per row and read
    # back as a test rather than as a number. Four of the words have the run
    # in them somewhere, and only two of those start with it.
    var got = answer(
        "SELECT count(*) AS c FROM words WHERE strpos(word, 'ap') > 0", "c"
    )
    same(got, [4], "c")


def test_a_trim_folds_the_way_a_character_count_folds() raises:
    # The shape that matters: a trim worked out per row and read back by the
    # length of what came off it.
    var got = answer(
        "SELECT length(trim(word)) AS c FROM padded WHERE n < 3", "c"
    )
    same(got, [2, 4], "c")


def test_a_substring_takes_the_characters_the_query_named() raises:
    # `words` is apple, apricot, banana, grape, the empty string, a null and
    # pineapple, so three from the front keeps a different set of letters for
    # each of them and leaves the last two alone.
    var got = cuts("SELECT substring(word, 1, 3) AS piece FROM words")
    assert_equal(len(got), 7, "one answer per row")
    assert_equal(got[0], "app", "the first")
    assert_equal(got[2], "ban", "and one from the middle")
    assert_equal(got[4], "", "the empty string has nothing to take")
    assert_equal(got[5], "null", "and a null stays a null")


def test_a_substring_with_no_length_runs_to_the_end() raises:
    var got = cuts("SELECT substring(word, 4) AS piece FROM words")
    assert_equal(got[0], "le", "what was left of a five letter word")
    assert_equal(got[6], "eapple", "and of a nine letter one")


def test_a_substr_is_the_same_function_under_duckdbs_other_name() raises:
    var got = cuts("SELECT substr(word, 2, 2) AS piece FROM words")
    assert_equal(got[0], "pp", "the first")
    assert_equal(got[6], "in", "and the last")


def test_the_keyword_spelling_reads_the_same_two_numbers() raises:
    var got = cuts("SELECT SUBSTRING(word FROM 2 FOR 2) AS piece FROM words")
    assert_equal(got[0], "pp", "the first")
    assert_equal(got[6], "in", "and the last")


def test_the_keyword_spelling_without_a_from_starts_at_the_first_letter() raises:
    var got = cuts("SELECT SUBSTRING(word FOR 3) AS piece FROM words")
    assert_equal(got[0], "app", "the first")
    assert_equal(got[6], "pin", "and the last")


def test_a_substring_in_a_where_reads_the_cut_column() raises:
    same(
        answer("SELECT n FROM words WHERE substring(word, 1, 2) = 'ap'", "n"),
        [1, 2],
        "n",
    )


def test_a_substring_whose_start_is_a_column_is_refused() raises:
    with assert_raises(contains="have to be written out"):
        _ = run("SELECT substring(word, n, 2) FROM words", session())


def test_a_substring_of_a_number_is_refused() raises:
    with assert_raises(contains="'substring' reads text"):
        _ = run("SELECT substring(n, 1, 2) FROM words", session())


def test_a_coalesce_fills_the_gaps_from_the_second_argument() raises:
    same(
        read_back(
            run("SELECT coalesce(mark, 99) AS m FROM gappy", session()), "m"
        ),
        [4, 4, 99, 9, 99, 1],
        "m",
    )


def test_a_coalesce_reads_its_arguments_in_the_order_written() raises:
    # The middle one is a null and fills nothing, so the third is what the gaps
    # come from, and a version that stopped at the first fallback would answer
    # a column that still had two gaps in it.
    same(
        read_back(
            run("SELECT coalesce(mark, NULL, 7) AS m FROM gappy", session()),
            "m",
        ),
        [4, 4, 7, 9, 7, 1],
        "m",
    )


def test_a_coalesce_of_one_argument_is_that_argument() raises:
    same(
        gapped(run("SELECT coalesce(mark) AS m FROM gappy", session()), "m"),
        [4, 4, -1, 9, -1, 1],
        "m",
    )


def test_an_ifnull_is_a_coalesce_of_two() raises:
    same(
        read_back(
            run("SELECT ifnull(mark, 0) AS m FROM gappy", session()), "m"
        ),
        [4, 4, 0, 9, 0, 1],
        "m",
    )


def test_a_nullif_takes_the_value_out_where_the_two_agree() raises:
    # The two nulls stay nulls. `mark = 4` is null on those rows rather than
    # false, the conditional takes its else side, and the else side is `mark`.
    same(
        gapped(run("SELECT nullif(mark, 4) AS m FROM gappy", session()), "m"),
        [-1, -1, -1, 9, -1, 1],
        "m",
    )


def test_a_coalesce_in_a_where_reads_the_filled_column() raises:
    same(
        answer("SELECT mark FROM gappy WHERE coalesce(mark, 0) > 3", "mark"),
        [4, 4, 9],
        "mark",
    )


def test_a_coalesce_moves_both_sides_to_the_type_they_agree_on() raises:
    # The column holds whole numbers and the fallback does not, so the answer is
    # the wider of the two and the column is what moves. The fallback is cast
    # rather than written as a fraction because the decimal literal is still
    # refused, which is a gap of its own and not this one.
    var out = run(
        "SELECT coalesce(mark, CAST(0 AS DOUBLE)) AS m FROM gappy", session()
    )

    assert_true(
        out.schema[0].dtype == LogicalType.FLOAT64, "the wider of the two"
    )
    var col = out.column("m").as_typed[DType.float64]()
    assert_equal(len(col), 6, "one answer per row")
    assert_equal(col[0], 4.0, "the value that was already there")
    assert_equal(col[2], 0.0, "and the gap taken from the fallback")


def test_a_coalesce_fills_a_column_of_text() raises:
    same(
        answer("SELECT n FROM words WHERE coalesce(word, 'zz') = 'zz'", "n"),
        [6],
        "n",
    )


def test_a_coalesce_whose_arguments_do_not_agree_is_refused() raises:
    with assert_raises(contains="have to agree on a type"):
        _ = run("SELECT coalesce(mark, 'a') FROM gappy", session())


def test_a_chain_of_ors_folds_left_to_right_and_keeps_every_arm() raises:
    same(
        answer(
            "SELECT qty FROM sales WHERE qty = 3 OR qty = 25 OR qty > 29",
            "qty",
        ),
        [3, 40, 25, 30],
        "qty",
    )


def test_an_and_nested_under_an_or_is_not_split_into_filters() raises:
    # The conjunction here cannot become a line of filters, because it only has
    # to hold on the rows the disjunction did not already keep, so this is the
    # one shape where an AND reaches the connective operator.
    same(
        answer(
            "SELECT qty FROM sales WHERE qty = 1 OR (qty > 10 AND shop = 1)",
            "qty",
        ),
        [12, 25, 1, 30],
        "qty",
    )


def test_a_boolean_expression_in_a_select_list_is_a_column() raises:
    same(
        truths(
            run("SELECT qty > 10 AND shop = 1 AS big FROM sales", session()),
            "big",
        ),
        [0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        "big",
    )


def test_a_not_in_a_select_list_turns_the_column_over() raises:
    same(
        truths(
            run("SELECT NOT (qty > 10) AS small FROM sales", session()),
            "small",
        ),
        [1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        "small",
    )


def test_a_cast_of_a_column_leaves_the_column_it_read_alone() raises:
    # The converted column is a column of its own, so qty is still the int64
    # every other expression in the query was bound against.
    var out = run(
        "SELECT qty, CAST(qty AS DOUBLE) AS wide FROM sales", session()
    )

    assert_equal(len(out.schema), 2, "two columns")
    assert_true(out.schema[0].dtype == LogicalType.INT64, "qty as it was")
    assert_true(out.schema[1].dtype == LogicalType.FLOAT64, "and the cast")
    same(read_back(out, "qty"), [5, 20, 3, 40, 12, 8, 25, 1, 30, 15], "qty")

    var wide = out.column("wide").as_typed[DType.float64]()
    assert_equal(wide[0], 5.0, "the first row converted")
    assert_equal(wide[6], 25.0, "and one from the middle chunk")


def test_a_cast_narrows_a_column_to_the_type_the_query_named() raises:
    var out = run("SELECT CAST(qty AS SMALLINT) AS small FROM sales", session())

    assert_true(out.schema[0].dtype == LogicalType.INT16, "int16")
    var col = out.column("small").as_typed[DType.int16]()
    assert_equal(len(col), 10, "every row")
    assert_equal(col[3], 40, "the value came across")


def test_a_cast_to_varchar_writes_the_numbers_out() raises:
    var out = run(
        "SELECT CAST(qty AS VARCHAR) AS written FROM sales", session()
    )

    assert_true(out.schema[0].dtype == LogicalType.STRING, "text")
    var col = out.column("written").as_strings()
    assert_equal(col[0], "5", "the first")
    assert_equal(col[3], "40", "and a two digit one")


def test_a_cast_of_an_expression_converts_what_the_expression_made() raises:
    var out = run(
        "SELECT CAST(qty * price AS DOUBLE) AS total FROM sales", session()
    )

    assert_true(out.schema[0].dtype == LogicalType.FLOAT64, "float64")
    var col = out.column("total").as_typed[DType.float64]()
    assert_equal(col[0], 50.0, "the first product")
    assert_equal(col[9], 90.0, "and the last")


def test_a_cast_in_a_where_runs_before_the_rows_are_kept() raises:
    same(
        answer(
            "SELECT qty FROM sales WHERE CAST(qty AS DOUBLE) > 20",
            "qty",
        ),
        [40, 25, 30],
        "qty",
    )


def test_a_cast_of_a_double_to_an_integer_rounds_the_way_duckdb_does() raises:
    # Halving the quantities gives five values that land on a half and five
    # that do not, and DuckDB 1.5.1 answers this query 2, 10, 2, 20, 6, 4, 12,
    # 0, 15, 8. Truncating would say 1 where it says 2 and 7 where it says 8.
    # The ties go to the even number, so 2.5 is 2 and 7.5 is 8, which is why
    # this is rounding and not adding a half first.
    same(
        answer(
            "SELECT CAST(CAST(qty AS DOUBLE) / 2 AS BIGINT) AS half FROM sales",
            "half",
        ),
        [2, 10, 2, 20, 6, 4, 12, 0, 15, 8],
        "half",
    )


def test_a_cast_of_a_double_rounds_whatever_integer_it_is_asked_for() raises:
    # The flag is set from the target type, so every integer width gets it and
    # not just the one the first test happened to name.
    var out = run(
        "SELECT CAST(CAST(qty AS DOUBLE) / 2 AS SMALLINT) AS half FROM sales",
        session(),
    )

    assert_true(out.schema[0].dtype == LogicalType.INT16, "int16")
    var col = out.column("half").as_typed[DType.int16]()
    assert_equal(col[2], 2, "1.5 rounded up")
    assert_equal(col[9], 8, "and 7.5 did too")


def test_a_cast_of_a_double_to_a_double_keeps_the_fraction() raises:
    # Nothing rounds on the way to a type that can hold what it is given, and
    # the flag the SQL side sets for an integer target is not set here at all.
    var out = run(
        "SELECT CAST(CAST(qty AS DOUBLE) / 2 AS DOUBLE) AS half FROM sales",
        session(),
    )

    var col = out.column("half").as_typed[DType.float64]()
    assert_equal(col[0], 2.5, "the first is still a half")
    assert_equal(col[9], 7.5, "and so is the last")


def test_a_cast_to_a_type_the_engine_has_no_column_for_says_so() raises:
    with assert_raises(contains="integers stop at 64 bits"):
        _ = run("SELECT CAST(qty AS HUGEINT) FROM sales", session())
    with assert_raises(contains="no exact decimal"):
        _ = run("SELECT CAST(qty AS DECIMAL(9,2)) FROM sales", session())
    with assert_raises(contains="TRY_CAST"):
        _ = run("SELECT TRY_CAST(qty AS BIGINT) FROM sales", session())


def test_a_window_over_the_whole_table_is_on_every_row() raises:
    # The same 159 the aggregate test asks for, except that here it arrives
    # beside the ten rows rather than instead of them.
    same(
        answer("SELECT qty, SUM(qty) OVER () AS total FROM sales", "total"),
        [159, 159, 159, 159, 159, 159, 159, 159, 159, 159],
        "total",
    )


def test_a_window_partitions_and_each_row_reads_its_own() raises:
    # The shops alternate, so the two totals alternate with them, and the rows
    # stay in the order they were read in rather than being gathered by shop.
    var out = run(
        "SELECT shop, SUM(qty) OVER (PARTITION BY shop) AS total FROM sales",
        session(),
    )
    same(read_back(out, "shop"), [1, 2, 1, 2, 1, 2, 1, 2, 1, 2], "shop")
    same(
        read_back(out, "total"),
        [75, 84, 75, 84, 75, 84, 75, 84, 75, 84],
        "total",
    )


def test_a_window_counts_the_rows_it_partitions_over() raises:
    same(
        answer("SELECT COUNT(*) OVER (PARTITION BY shop) AS n FROM sales", "n"),
        [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        "n",
    )


def test_a_qualify_keeps_the_rows_the_window_says_to() raises:
    # Shop 2 totals 84 and shop 1 totals 75, so the bound keeps one shop, and
    # it keeps every row of it rather than one row standing for the group.
    same(
        answer(
            (
                "SELECT qty FROM sales QUALIFY SUM(qty) OVER (PARTITION BY"
                " shop) > 80"
            ),
            "qty",
        ),
        [20, 40, 8, 1, 15],
        "qty",
    )


def test_a_where_under_a_window_changes_what_the_window_reduces() raises:
    # The filter runs first, so the total is over what survived it and not over
    # the table, which is the difference between a WHERE and a QUALIFY.
    same(
        answer(
            "SELECT qty, SUM(qty) OVER () AS total FROM sales WHERE qty > 20",
            "total",
        ),
        [95, 95, 95],
        "total",
    )


def test_a_running_window_is_refused_by_name() raises:
    with assert_raises(contains="OVER an ORDER BY"):
        _ = run("SELECT SUM(qty) OVER (ORDER BY qty) FROM sales", session())


def test_a_query_may_read_a_subquery_where_a_table_goes() raises:
    same(
        answer(
            "SELECT total FROM (SELECT qty * price AS total FROM sales) v",
            "total",
        ),
        [50, 40, 21, 40, 60, 72, 75, 100, 120, 90],
        "total",
    )


def test_the_outer_query_filters_what_the_subquery_handed_out() raises:
    # The filter is written over the alias the subquery invented, which is a
    # name the query inside it produced and the table underneath does not have.
    same(
        answer(
            (
                "SELECT total FROM (SELECT qty * price AS total FROM sales) v"
                " WHERE total > 80"
            ),
            "total",
        ),
        [100, 120, 90],
        "total",
    )


def test_a_column_of_a_derived_table_may_be_written_with_its_name() raises:
    same(
        answer(
            (
                "SELECT v.total FROM (SELECT qty * price AS total FROM sales) v"
                " WHERE v.total > 100"
            ),
            "total",
        ),
        [120],
        "total",
    )


def test_an_aggregate_inside_a_subquery_folds_before_the_outer_query() raises:
    # The group by runs inside and the outer query filters the answers it
    # produced, which is the shape a HAVING has and the shape a query uses when
    # it wants to filter on something a HAVING cannot say.
    var out = run(
        (
            "SELECT shop, total FROM (SELECT shop, SUM(qty) AS total FROM sales"
            " GROUP BY shop) v WHERE total > 80 ORDER BY shop"
        ),
        session(),
    )
    same(read_back(out, "shop"), [2], "shop")
    same(read_back(out, "total"), [84], "total")


def test_a_subquery_may_be_joined_to_a_table() raises:
    # The subquery is on the left because a join whose right input arrives in
    # more than one chunk raises out of the operator, which is #583 and is the
    # same with two plain tables.
    var out = run(
        (
            "SELECT band, total FROM (SELECT qty, qty * price AS total FROM"
            " sales) v JOIN tiers ON v.qty = tiers.band ORDER BY band"
        ),
        session(),
    )
    same(read_back(out, "band"), [3, 20, 40], "band")
    same(read_back(out, "total"), [21, 40, 40], "total")


def test_a_subquery_inside_a_subquery_runs_too() raises:
    same(
        answer(
            (
                "SELECT total FROM (SELECT total FROM (SELECT qty * price AS"
                " total FROM sales) inner_v WHERE total > 90) v"
            ),
            "total",
        ),
        [100, 120],
        "total",
    )


def test_the_column_aliases_on_a_derived_table_run() raises:
    # The list renames what the subquery produced, so the outer query reads
    # `worth` and the name the subquery gave the column is gone.
    same(
        answer(
            (
                "SELECT worth FROM (SELECT qty * price AS total FROM sales)"
                " v(worth) WHERE worth > 90"
            ),
            "worth",
        ),
        [100, 120],
        "worth",
    )


def test_a_short_alias_list_on_a_derived_table_leaves_the_rest_alone() raises:
    # Two columns and one name, so the first is renamed and the second keeps
    # what it had, which is the prefix rule.
    var out = run(
        (
            "SELECT much, price FROM (SELECT qty, price FROM sales) v(much)"
            " WHERE much > 25"
        ),
        session(),
    )
    same(read_back(out, "much"), [40, 30], "much")
    same(read_back(out, "price"), [1, 4], "price")


def test_more_aliases_than_a_derived_table_produces_is_refused() raises:
    with assert_raises(contains="has 1 columns available but 2 columns"):
        _ = run("SELECT a FROM (SELECT qty FROM sales) v(a, b)", session())


def test_a_with_hands_its_rows_to_the_query_that_names_it() raises:
    same(
        answer(
            (
                "WITH big AS (SELECT qty FROM sales WHERE qty > 20)"
                " SELECT qty FROM big"
            ),
            "qty",
        ),
        [40, 25, 30],
        "qty",
    )


def test_a_cte_read_twice_answers_the_same_both_times() raises:
    same(
        answer(
            (
                "WITH big AS (SELECT qty FROM sales WHERE qty > 20)"
                " SELECT qty FROM big UNION ALL SELECT qty FROM big"
            ),
            "qty",
        ),
        [40, 25, 30, 40, 25, 30],
        "qty",
    )


def test_a_cte_may_read_the_one_bound_before_it() raises:
    same(
        answer(
            (
                "WITH bigger AS (SELECT qty FROM sales WHERE qty > 10),"
                " fewer AS (SELECT qty FROM bigger WHERE qty < 30)"
                " SELECT qty FROM fewer"
            ),
            "qty",
        ),
        [20, 12, 25, 15],
        "qty",
    )


def test_the_alias_list_on_a_cte_renames_what_it_hands_out() raises:
    same(
        answer(
            (
                "WITH v(n) AS (SELECT qty, price FROM sales WHERE qty > 25)"
                " SELECT n FROM v"
            ),
            "n",
        ),
        [40, 30],
        "n",
    )


def test_a_cte_may_fold_and_the_query_reads_the_answer() raises:
    same(
        answer(
            (
                "WITH per AS (SELECT shop, sum(qty) AS total FROM sales"
                " GROUP BY shop) SELECT total FROM per ORDER BY total"
            ),
            "total",
        ),
        [75, 84],
        "total",
    )


def test_a_cte_may_be_joined_to_a_table() raises:
    # The CTE is on the left for the reason the subquery above it is, which is
    # #583.
    var out = run(
        (
            "WITH v AS (SELECT qty, qty * price AS total FROM sales)"
            " SELECT band, total FROM v JOIN tiers ON v.qty = tiers.band"
            " ORDER BY band"
        ),
        session(),
    )
    same(read_back(out, "band"), [3, 20, 40], "band")
    same(read_back(out, "total"), [21, 40, 40], "total")


def test_a_recursive_cte_is_refused_by_name() raises:
    with assert_raises(contains="recursive CTE"):
        _ = run(
            (
                "WITH RECURSIVE n(i) AS (SELECT 1 AS i UNION ALL"
                " SELECT i + 1 FROM n WHERE i < 5) SELECT i FROM n"
            ),
            session(),
        )


def test_two_tables_that_share_a_name_join_on_that_name() raises:
    # The star writes the shared name twice here, once from each side, which is
    # what DuckDB writes for the same query and is the whole reason the operator
    # is told its output by position.
    var out = run(
        (
            "SELECT * FROM sales JOIN shops ON sales.shop = shops.shop"
            " ORDER BY qty"
        ),
        session(),
    )
    assert_equal(len(out.schema), 5, "five columns")
    assert_equal(out.schema[2].name, "shop", "the left one")
    assert_equal(out.schema[3].name, "shop", "and the right one under its name")
    same(read_back(out, "qty"), [1, 3, 5, 8, 12, 15, 20, 25, 30, 40], "qty")
    same(read_back(out, "floor"), [22, 11, 11, 22, 11, 22, 22, 11, 11, 22], "f")


def test_a_using_join_runs_and_writes_the_pair_once() raises:
    var out = run(
        "SELECT qty, floor FROM sales JOIN shops USING (shop) ORDER BY qty",
        session(),
    )
    same(read_back(out, "qty"), [1, 3, 5, 8, 12, 15, 20, 25, 30, 40], "qty")
    same(read_back(out, "floor"), [22, 11, 11, 22, 11, 22, 22, 11, 11, 22], "f")

    var stars = run("SELECT * FROM sales NATURAL JOIN shops", session())
    assert_equal(len(stars.schema), 4, "the shared name written once")
    assert_equal(stars.schema[2].name, "shop")
    assert_equal(stars.schema[3].name, "floor")
    same(
        read_back(stars, "floor"), [11, 22, 11, 22, 11, 22, 11, 22, 11, 22], "f"
    )


def test_a_semi_join_keeps_the_left_rows_that_matched() raises:
    var out = run(
        (
            "SELECT qty, price FROM sales SEMI JOIN tiers ON qty = band"
            " ORDER BY qty"
        ),
        session(),
    )
    assert_equal(len(out.schema), 2, "the right side hands out no column")
    same(read_back(out, "qty"), [3, 20, 40], "qty")
    same(read_back(out, "price"), [7, 2, 1], "price")


def test_an_anti_join_keeps_the_left_rows_that_did_not() raises:
    # The other half of the same ten rows, which is the property worth having:
    # a semi and an anti join over one condition partition the left side.
    var out = run(
        "SELECT qty FROM sales ANTI JOIN tiers ON qty = band ORDER BY qty",
        session(),
    )
    same(read_back(out, "qty"), [1, 5, 8, 12, 15, 25, 30], "qty")


def test_an_outer_join_takes_a_condition_about_its_right_side() raises:
    # Three bands charge more than 250 and band 20 is not one of them, so the
    # row that band would have matched is padded instead. That is the whole of
    # it: the condition throws away right rows before the pairing and the left
    # rows come through either way, which is what an outer join promises.
    var out = run(
        (
            "SELECT qty, rate FROM sales LEFT JOIN tiers ON qty = band AND rate"
            " > 250 ORDER BY qty"
        ),
        session(),
    )
    same(read_back(out, "qty"), [1, 3, 5, 8, 12, 15, 20, 25, 30, 40], "qty")
    same(
        gapped(out, "rate"),
        [-1, 300, -1, -1, -1, -1, -1, -1, -1, 400],
        "rate",
    )


def test_a_semi_and_an_anti_join_take_one_too() raises:
    # The same condition in the same place over the two joins that ask the
    # right side whether a row matched. Bands 3 and 40 are the ones left after
    # it, so the two answers are the ten rows split on those.
    same(
        answer(
            (
                "SELECT qty FROM sales SEMI JOIN tiers ON qty = band AND rate >"
                " 250 ORDER BY qty"
            ),
            "qty",
        ),
        [3, 40],
        "qty",
    )
    same(
        answer(
            (
                "SELECT qty FROM sales ANTI JOIN tiers ON qty = band AND rate >"
                " 250 ORDER BY qty"
            ),
            "qty",
        ),
        [1, 5, 8, 12, 15, 20, 25, 30],
        "qty",
    )


def test_an_outer_condition_about_the_left_side_is_still_refused() raises:
    # A left row that fails this one is still a row the join has to answer for,
    # padded rather than dropped, and there is nowhere to test it that keeps it.
    with assert_raises(contains="left join on equalities"):
        _ = run(
            "SELECT qty FROM sales LEFT JOIN tiers ON qty = band AND price > 5",
            session(),
        )


def test_a_semi_join_writes_a_left_row_once_however_many_matched() raises:
    # Band 3 is in the dupes frame twice. An inner join would answer with two
    # rows here and a semi join answers whether there was a match at all, so the
    # count is what tells the two apart.
    var out = run(
        "SELECT qty FROM sales SEMI JOIN dupes ON qty = band ORDER BY qty",
        session(),
    )
    same(read_back(out, "qty"), [3, 20], "qty")


def test_a_star_over_a_semi_join_writes_the_left_side_alone() raises:
    var out = run("SELECT * FROM sales SEMI JOIN shops USING (shop)", session())
    assert_equal(len(out.schema), 3, "the sales columns and nothing else")
    assert_equal(out.schema[0].name, "qty")
    assert_equal(out.schema[1].name, "price")
    assert_equal(out.schema[2].name, "shop")


def test_the_right_side_of_a_semi_join_cannot_be_read_above_it() raises:
    # DuckDB refuses the same query, and for the same reason: the join answered
    # a question about that table rather than joining it, so above the node
    # there is no such table to name.
    with assert_raises(contains="shops"):
        _ = run(
            "SELECT shops.floor FROM sales SEMI JOIN shops USING (shop)",
            session(),
        )


def test_an_in_over_a_subquery_runs_as_a_semi_join() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE qty IN (SELECT band FROM tiers)"
                " ORDER BY qty"
            ),
            "qty",
        ),
        [3, 20, 40],
        "qty",
    )


def test_an_in_answers_a_row_once_however_many_matched_it() raises:
    # Band 3 is in the dupes frame twice, and `IN` asks whether a value is in a
    # set rather than how many times.
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE qty IN (SELECT band FROM dupes)"
                " ORDER BY qty"
            ),
            "qty",
        ),
        [3, 20],
        "qty",
    )


def test_the_rest_of_the_where_still_holds_beside_an_in() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE price > 4"
                " AND qty IN (SELECT band FROM tiers) ORDER BY qty"
            ),
            "qty",
        ),
        [3],
        "qty",
    )


def test_a_not_in_over_a_subquery_keeps_the_rows_that_matched_nothing() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE qty NOT IN"
                " (SELECT band FROM tiers) ORDER BY qty"
            ),
            "qty",
        ),
        [1, 5, 8, 12, 15, 25, 30],
        "qty",
    )


def test_a_not_in_over_a_subquery_holding_a_null_keeps_nothing() raises:
    # The answer everyone gets wrong and DuckDB gets right. A row that matched
    # nothing might have matched the null, so it is null rather than true, the
    # NOT over it stays null, and a WHERE keeps a row on true.
    assert_equal(
        len(
            run(
                (
                    "SELECT qty FROM sales WHERE qty NOT IN"
                    " (SELECT band FROM gaps)"
                ),
                session(),
            )
        ),
        0,
    )


def test_an_in_written_in_the_select_list_answers_on_every_row() raises:
    same(
        truths(
            run(
                "SELECT qty IN (SELECT band FROM tiers) AS hit FROM sales",
                session(),
            ),
            "hit",
        ),
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        "hit",
    )


def test_an_in_over_a_subquery_holding_a_null_is_null_where_it_missed() raises:
    same(
        truths(
            run(
                "SELECT qty IN (SELECT band FROM gaps) AS hit FROM sales",
                session(),
            ),
            "hit",
        ),
        [-1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
        "hit",
    )


def test_a_not_in_written_in_the_select_list_is_that_negated() raises:
    same(
        truths(
            run(
                "SELECT qty NOT IN (SELECT band FROM gaps) AS gone FROM sales",
                session(),
            ),
            "gone",
        ),
        [-1, 0, 0, -1, -1, -1, -1, -1, -1, -1],
        "gone",
    )


def test_an_in_under_an_or_keeps_what_either_side_keeps() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE price > 4 OR qty IN"
                " (SELECT band FROM tiers) ORDER BY qty"
            ),
            "qty",
        ),
        [1, 3, 5, 8, 12, 15, 20, 40],
        "qty",
    )


def test_an_equals_any_keeps_what_an_in_keeps() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE qty = ANY"
                " (SELECT band FROM tiers) ORDER BY qty"
            ),
            "qty",
        ),
        [3, 20, 40],
        "qty",
    )


def test_a_not_equals_all_over_a_null_keeps_nothing() raises:
    # The same null aware answer a NOT IN gets, because it is the same lowering
    # and there is nothing written for this case anywhere.
    assert_equal(
        len(
            run(
                (
                    "SELECT qty FROM sales WHERE qty <> ALL"
                    " (SELECT band FROM gaps)"
                ),
                session(),
            )
        ),
        0,
    )


def test_an_equals_any_answers_on_every_row() raises:
    same(
        truths(
            run(
                "SELECT qty = ANY (SELECT band FROM tiers) AS hit FROM sales",
                session(),
            ),
            "hit",
        ),
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        "hit",
    )


def test_a_greater_than_any_keeps_what_beat_the_smallest_row() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE qty > ANY"
                " (SELECT band FROM tiers) ORDER BY qty"
            ),
            "qty",
        ),
        [5, 8, 12, 15, 20, 25, 30, 40],
        "qty",
    )


def test_a_greater_than_any_answers_on_every_row() raises:
    same(
        truths(
            run(
                "SELECT qty > ANY (SELECT band FROM tiers) AS over FROM sales",
                session(),
            ),
            "over",
        ),
        [1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
        "over",
    )


def test_a_greater_or_equal_all_is_false_where_nothing_reaches_the_top() raises:
    # The largest band is 99 and no sale reaches it, so every row is false and
    # none of them is null, because the subquery holds no null.
    same(
        truths(
            run(
                "SELECT qty >= ALL (SELECT band FROM tiers) AS top FROM sales",
                session(),
            ),
            "top",
        ),
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "top",
    )


def test_a_less_than_all_is_true_only_under_the_smallest_row() raises:
    same(
        truths(
            run(
                "SELECT qty < ALL (SELECT band FROM tiers) AS under FROM sales",
                session(),
            ),
            "under",
        ),
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        "under",
    )


def test_a_not_equals_any_is_true_wherever_the_two_ends_differ() raises:
    same(
        truths(
            run(
                (
                    "SELECT qty <> ANY (SELECT band FROM tiers) AS other"
                    " FROM sales"
                ),
                session(),
            ),
            "other",
        ),
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "other",
    )


def test_an_equals_all_is_false_over_a_subquery_of_several_rows() raises:
    same(
        truths(
            run(
                "SELECT qty = ALL (SELECT band FROM tiers) AS only FROM sales",
                session(),
            ),
            "only",
        ),
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "only",
    )


def test_a_null_in_the_subquery_turns_a_false_into_a_null() raises:
    # The answer this shape is for. A row that beat every band there was to
    # beat did not beat the null, so it is neither true nor false, while a row
    # that lost to a band it could see is false whatever the null was.
    same(
        truths(
            run(
                "SELECT qty > ALL (SELECT band FROM gaps) AS over FROM sales",
                session(),
            ),
            "over",
        ),
        [0, 0, 0, -1, 0, 0, -1, 0, -1, 0],
        "over",
    )


def test_a_null_in_the_subquery_leaves_a_true_alone() raises:
    same(
        truths(
            run(
                "SELECT qty > ANY (SELECT band FROM gaps) AS over FROM sales",
                session(),
            ),
            "over",
        ),
        [1, 1, -1, 1, 1, 1, 1, -1, 1, 1],
        "over",
    )


def test_a_quantified_comparison_over_no_rows_is_the_quantifier() raises:
    # `ANY` over nothing is false and `ALL` over nothing is true, whatever is
    # on the other side of the comparison, and the fold hands out the one row
    # that says the subquery was empty.
    same(
        truths(
            run(
                (
                    "SELECT qty > ANY (SELECT band FROM tiers WHERE band >"
                    " 1000) AS over FROM sales"
                ),
                session(),
            ),
            "over",
        ),
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "over",
    )
    same(
        truths(
            run(
                (
                    "SELECT qty > ALL (SELECT band FROM tiers WHERE band >"
                    " 1000) AS over FROM sales"
                ),
                session(),
            ),
            "over",
        ),
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "over",
    )


def test_a_where_keeps_a_row_on_true_and_not_on_a_null() raises:
    assert_equal(
        len(
            run(
                "SELECT qty FROM sales WHERE qty > ALL (SELECT band FROM gaps)",
                session(),
            )
        ),
        0,
    )


def test_an_in_over_a_subquery_that_kept_no_rows_keeps_no_rows() raises:
    # The build side is a column no rows reached, which has no chunks at all
    # rather than one empty chunk, and the join used to raise on that. #611.
    assert_equal(
        len(
            run(
                (
                    "SELECT qty FROM sales WHERE qty IN"
                    " (SELECT band FROM tiers WHERE band > 1000)"
                ),
                session(),
            )
        ),
        0,
    )


def test_an_in_over_a_subquery_that_kept_no_rows_is_false_as_a_value() raises:
    # False rather than null, since there is nothing to match and no null in
    # what was not matched against.
    same(
        truths(
            run(
                (
                    "SELECT qty IN (SELECT band FROM tiers WHERE band > 1000)"
                    " AS hit FROM sales"
                ),
                session(),
            ),
            "hit",
        ),
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "hit",
    )


def test_a_not_in_over_a_subquery_that_kept_no_rows_keeps_them_all() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE qty NOT IN"
                " (SELECT band FROM tiers WHERE band > 1000) ORDER BY qty"
            ),
            "qty",
        ),
        [1, 3, 5, 8, 12, 15, 20, 25, 30, 40],
        "qty",
    )


def test_a_correlated_exists_runs_as_a_semi_join() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE EXISTS"
                " (SELECT 1 FROM tiers WHERE tiers.band = sales.qty)"
                " ORDER BY qty"
            ),
            "qty",
        ),
        [3, 20, 40],
        "qty",
    )


def test_a_correlated_not_exists_runs_as_the_anti_join() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE NOT EXISTS"
                " (SELECT 1 FROM tiers WHERE tiers.band = sales.qty)"
                " ORDER BY qty"
            ),
            "qty",
        ),
        [1, 5, 8, 12, 15, 25, 30],
        "qty",
    )


def test_the_uncorrelated_half_of_an_exists_still_holds() raises:
    # Band 20 is a tier and its rate is under the bar, so the row the first of
    # these tests kept for it goes.
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE EXISTS"
                " (SELECT 1 FROM tiers WHERE tiers.band = sales.qty"
                " AND tiers.rate > 250) ORDER BY qty"
            ),
            "qty",
        ),
        [3, 40],
        "qty",
    )


def test_an_exists_answers_a_row_once_however_many_matched_it() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE EXISTS"
                " (SELECT 1 FROM dupes WHERE dupes.band = sales.qty)"
                " ORDER BY qty"
            ),
            "qty",
        ),
        [3, 20],
        "qty",
    )


def test_an_exists_correlated_on_a_column_that_repeats() raises:
    # The key is the shop rather than the quantity, so several outer rows share
    # a match and each is still written once.
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE EXISTS"
                " (SELECT 1 FROM shops WHERE shops.shop = sales.shop"
                " AND shops.floor = 22) ORDER BY qty"
            ),
            "qty",
        ),
        [1, 8, 15, 20, 40],
        "qty",
    )


def test_the_rest_of_the_where_still_holds_beside_an_exists() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE price > 4 AND EXISTS"
                " (SELECT 1 FROM tiers WHERE tiers.band = sales.qty)"
                " ORDER BY qty"
            ),
            "qty",
        ),
        [3],
        "qty",
    )


def test_an_uncorrelated_exists_keeps_every_row_or_none() raises:
    # It asks whether the table has any row at all, which every outer row gets
    # the same answer to, so it is counted under a cross join rather than joined
    # on. The table has rows, so every row is kept.
    assert_equal(
        len(
            run(
                "SELECT qty FROM sales WHERE EXISTS (SELECT 1 FROM tiers)",
                session(),
            )
        ),
        10,
    )


def test_a_subquery_that_answers_one_value_runs() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE qty > (SELECT min(band) FROM"
                " tiers) ORDER BY qty"
            ),
            "qty",
        ),
        [5, 8, 12, 15, 20, 25, 30, 40],
        "qty",
    )


def test_two_subqueries_in_one_where_both_run() raises:
    same(
        answer(
            (
                "SELECT qty FROM sales WHERE qty > (SELECT min(band) FROM"
                " tiers) AND price < (SELECT min(band) FROM tiers) ORDER BY qty"
            ),
            "qty",
        ),
        [20, 40],
        "qty",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
