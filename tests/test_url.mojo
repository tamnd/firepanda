"""Tests for the hand written hostname extractor and for the query it is for.

The extractor stands in for a regular expression nobody has written yet, so the
tests are mostly the cases where a reasonable person would write something
simpler than the regex and get a different answer: a `www.` that is the whole
hostname, a URL with no path at all, an uppercase scheme, a newline in the path
and a newline in the host. Each of those is a place where stripping a prefix and
cutting at the first slash diverges from what `regexp_replace` does, and each of
them is a row that would quietly change group when RE2 lands.

The last test is q28 itself, without the regex. It is the only query in
ClickBench whose group by key is computed rather than read, and the point of
writing it now is that the shape underneath it, a group by on a derived text
column with an average, a count and a smallest string, a having on the count and
an ordered limit over the result, is the part that can be wrong. The extractor is
one call inside it and the day there is a regex engine it is a different call.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.strings import (
    StringArray,
    StringBuilder,
    strings_from_list,
)
from firepanda.array.value import Value
from firepanda.exec.morsel import MORSEL_ROWS
from firepanda.frame.frame import DataFrame
from firepanda.frame.groupby import AggSpec
from firepanda.frame.series import Series
from firepanda.kernel.binary import BinaryOp, binary_value_any
from firepanda.kernel.group import AggKind
from firepanda.kernel.substr import text_byte_length
from firepanda.kernel.url import text_hostname


def hosts(values: List[String]) raises -> List[String]:
    """Extracts every hostname and hands the answers back as strings.

    Args:
        values: The elements of the column to build.

    Returns:
        One answer per input, in order.

    Raises:
        Error: Only what the kernel raises.
    """
    var got = text_hostname(strings_from_list(values))
    var out = List[String](capacity=len(got))
    for i in range(len(got)):
        out.append(got[i])
    return out^


def test_a_url_gives_back_its_hostname() raises:
    var got = hosts(
        [
            String("http://example.com/"),
            String("https://example.com/path/deeper?q=1"),
            String("http://www.example.com/a"),
            String("https://www.example.com/"),
        ]
    )
    assert_equal(got[0], "example.com")
    assert_equal(got[1], "example.com")
    assert_equal(got[2], "example.com")
    assert_equal(got[3], "example.com")


def test_a_string_the_pattern_cannot_read_comes_back_whole() raises:
    # Every one of these is a row `regexp_replace` hands back untouched, which
    # means it becomes its own group rather than disappearing.
    var got = hosts(
        [
            String(""),
            String("not a url at all"),
            String("ftp://example.com/a"),
            String("http://example.com"),
            String("HTTP://EXAMPLE.COM/a"),
            String("http://"),
            String("http:/example.com/a"),
        ]
    )
    assert_equal(got[0], "")
    assert_equal(got[1], "not a url at all")
    assert_equal(got[2], "ftp://example.com/a")
    # No path, so `[^/]+` has nothing to stop at and the pattern fails.
    assert_equal(got[3], "http://example.com")
    # The pattern is case sensitive and so is this.
    assert_equal(got[4], "HTTP://EXAMPLE.COM/a")
    # An empty host is not `[^/]+`.
    assert_equal(got[5], "http://")
    assert_equal(got[6], "http:/example.com/a")


def test_a_www_that_is_the_whole_host_is_not_stripped() raises:
    # The optional group is preferred, so the engine takes `www.` first, finds
    # nothing left before the slash, backs up and reads the host as `www.`.
    # Stripping the prefix unconditionally answers the empty string here.
    var got = hosts(
        [
            String("http://www./x"),
            String("http://www.www./y"),
            String("http://www/z"),
        ]
    )
    assert_equal(got[0], "www.")
    assert_equal(got[1], "www.")
    assert_equal(got[2], "www")


def test_a_newline_in_the_path_means_there_is_no_host() raises:
    # `.` does not match a newline and `$` is the end of the text, so the
    # pattern cannot read this at all and the whole string comes back.
    var got = hosts(
        [
            String("http://example.com/a\nb"),
            String("http://example.com/a\n"),
            String("http://ex\nample.com/a"),
        ]
    )
    assert_equal(got[0], "http://example.com/a\nb")
    assert_equal(got[1], "http://example.com/a\n")
    # A negated class is not `.`, so a newline inside the host is allowed.
    assert_equal(got[2], "ex\nample.com")


def test_a_null_stays_null_and_an_empty_column_answers_nothing() raises:
    var builder = StringBuilder(capacity=3)
    builder.append(String("http://example.com/a").as_bytes())
    builder.append_null()
    builder.append(String("").as_bytes())
    var got = text_hostname(builder^.finish())
    assert_equal(len(got), 3)
    assert_true(got.is_valid(0))
    assert_equal(got[0], "example.com")
    assert_false(got.is_valid(1))
    assert_true(got.is_valid(2))
    assert_equal(got[2], "")

    assert_equal(len(text_hostname(strings_from_list(List[String]()))), 0)


def test_a_hostname_too_long_for_a_view_goes_through_the_payload() raises:
    # Twelve bytes is everything a view holds, so a host either side of that
    # line takes a different route out of the kernel.
    var short = String("http://ab.co/x")
    var long = String("http://a-rather-long-hostname.example.com/x")
    var got = hosts([short, long, String("http://abcdefghijkl/x")])
    assert_equal(got[0], "ab.co")
    assert_equal(got[1], "a-rather-long-hostname.example.com")
    assert_equal(got[2], "abcdefghijkl")


def test_the_answer_does_not_change_across_a_morsel_boundary() raises:
    var n = MORSEL_ROWS + 1000
    var builder = StringBuilder(capacity=n)
    for i in range(n):
        if i % 4 == 0:
            builder.append_null()
        elif i % 4 == 1:
            builder.append(String("http://www.short.io/p").as_bytes())
        elif i % 4 == 2:
            builder.append(
                String("https://a-long-hostname.example.com/p").as_bytes()
            )
        else:
            builder.append(String("no scheme here").as_bytes())
    var got = text_hostname(builder^.finish())

    assert_equal(len(got), n)
    for i in range(n - 8, n):
        if i % 4 == 0:
            assert_false(got.is_valid(i))
        elif i % 4 == 1:
            assert_equal(got[i], "short.io")
        elif i % 4 == 2:
            assert_equal(got[i], "a-long-hostname.example.com")
        else:
            assert_equal(got[i], "no scheme here")


def test_the_rest_of_q28_runs_against_the_extractor() raises:
    """Runs q28's shape with the extractor where the regex will go.

    `SELECT REGEXP_REPLACE(Referer, ...) AS k, AVG(length(Referer)) AS l,
    COUNT(*) AS c, MIN(Referer) FROM hits WHERE Referer <> '' GROUP BY k
    HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25`, with the having and the
    limit scaled to a column this size.

    Raises:
        AssertionError: If any step of the shape answers something else.
    """
    var referers = List[String]()
    # Three rows of one host, with the longest URL last so the average is not
    # the first row's length.
    referers.append(String("http://www.long-host.example.com/a"))
    referers.append(String("https://long-host.example.com/bb"))
    referers.append(String("http://long-host.example.com/cccccccccc"))
    # Two rows of another, which the having drops.
    referers.append(String("http://short.io/a"))
    referers.append(String("http://www.short.io/b"))
    # Three rows that are not URLs, which group together as themselves.
    referers.append(String("direct"))
    referers.append(String("direct"))
    referers.append(String("direct"))
    # Two empty rows, which the where drops before any of this happens.
    referers.append(String(""))
    referers.append(String(""))

    var series = List[Series]()
    series.append(Series("Referer", strings_from_list(referers)))
    var hits = DataFrame.from_series(series^)

    var kept = hits.filter(
        binary_value_any(
            hits.column("Referer").values, Value(String("")), BinaryOp.NE
        ).as_typed[DType.bool]()
    )
    assert_equal(len(kept), 8)

    var text = kept.column("Referer").as_strings()
    var wide = List[Series]()
    wide.append(Series("k", text_hostname(text)))
    wide.append(Series("l", text_byte_length(text)))
    wide.append(Series("Referer", StringArray(copy=text)))
    var frame = DataFrame.from_series(wide^)

    var specs = List[AggSpec]()
    specs.append(AggSpec("l", AggKind.MEAN, "l"))
    specs.append(AggSpec("k", AggKind.COUNT, "c"))
    specs.append(AggSpec("Referer", AggKind.MIN, "m"))
    var grouped = frame.group_by(["k"], specs)
    assert_equal(len(grouped), 3)

    var having = grouped.filter(
        binary_value_any(
            grouped.column("c").values, Value(Int64(2)), BinaryOp.GT
        ).as_typed[DType.bool]()
    )
    assert_equal(len(having), 2)

    var answer = having.sort_limit(["l"], [True], [False], 25)
    assert_equal(len(answer), 2)

    var keys = answer.column("k")
    var means = answer.column("l").as_typed[DType.float64]()
    var counts = answer.column("c").as_typed[DType.int64]()
    var mins = answer.column("m")

    # The URLs are 34, 32 and 39 bytes, so the average is exactly 35.
    assert_equal(keys.text(0), "long-host.example.com")
    assert_equal(means[0], 35.0)
    assert_equal(counts[0], 3)
    assert_equal(mins.text(0), "http://long-host.example.com/cccccccccc")

    assert_equal(keys.text(1), "direct")
    assert_equal(means[1], 6.0)
    assert_equal(counts[1], 3)
    assert_equal(mins.text(1), "direct")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
