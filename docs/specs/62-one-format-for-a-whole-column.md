# One format for a whole column

## 1. A value does not know how it prints

Document 61 finished the spacing and left the contents of a float cell alone, and the contents were wrong in a way that is easy to miss because every single value in isolation looks right. `1e-5` printed here as `1e-05` and prints in pandas as `0.00001`. `[1234567.125, 2.0]` printed here as `1234567.125` beside `2.0` and prints in pandas as `1234567.125` beside `2.000`. The second one is the interesting case, because there is no rule you can apply to `2.0` on its own that produces `2.000`. The three zeros are there because of the number beside it.

That is the whole shape of this document. A float cell is not rendered from the float. It is rendered from the column, and the column decides one format and applies it to every value in it, so the same `2.0` prints as `2.0` in one frame and `2.000` in another and `2.000000e+00` in a third depending entirely on what it is sitting next to. Everything else here is the detail of how the column decides, and every rule in it was measured against a running pandas rather than reasoned out, because three separate plausible models were wrong before the fourth one survived.

## 2. Everything is written fixed first

The column starts by writing every value in fixed point at `display.precision` places, which is six by default. There is no shortest round trip formatting anywhere in this, and that alone is a change: Mojo prints a `Float64` at the shortest representation that reads back as the same value, which for one third is seventeen characters and makes a table unreadable, so a table was already rounding to six places. What is new is that six places is not the end of it, it is the first step of it, and the two steps that follow both look at the whole column.

The value is written with a space in front of it if it is not negative, which is the same place document 61 established, except that here it is inside the string rather than added by the layer above. That matters because the next two decisions measure string lengths and the place counts.

## 3. The zeros come off the column and not off the value

Once every cell is written, trailing zeros come off. They come off one place at a time, from every cell at once, and only while every cell in the column still ends in a zero. The moment one cell does not, the stripping stops for all of them.

So `[1.5, 2.25]` is written `1.500000` and `2.250000`, and four rounds of stripping take it to `1.50` and `2.25`, at which point `2.25` does not end in a zero and `1.50` keeps its zero forever. And `[1234567.125, 2.0]` is written `1234567.125000` and `2.000000`, and three rounds take it to `1234567.125` and `2.000`, which is where it stops. The three zeros on the two are not decoration, they are the places that the other value in the column needed.

At least one place always survives. A column of `[1.0, 2.0]` would strip all the way to `1.` and `2.` if nothing stopped it, so the last round puts a zero back and the answer is `1.0` and `2.0`.

A cell only takes part in this if it is a plain number, which means an optional minus, at least one digit, a point, and then digits. A `NaN`, an infinity, a null cell and anything already in scientific notation are all not plain numbers, so they neither block the stripping nor get stripped themselves. That is why `[0.0078125, 1.0]` and `[inf, 1.0]` behave completely differently despite both looking like a column with an awkward value in it.

## 4. The switch to an exponent is two rules at once

After the stripping, the column measures itself. It takes the longest cell it has, counting the place in front, and asks two questions. Is that longer than precision plus six, which is twelve at the default. And is there a value in the column whose magnitude is over a million. If both are true, the whole column is thrown away and rewritten in scientific notation at the same precision, and nothing is stripped off that.

Both halves are needed and neither is sufficient, which is why every simpler model failed. `123456789.0` is twelve characters with its place and stays fixed. `1234567890.0` is thirteen and goes to `1.234568e+09`. `-123456789.0` is also twelve, because the minus takes the place rather than adding to it, and also stays fixed. `-1234567890.0` is thirteen and goes. `1234567.125` is twelve and stays, and `1234567.0625` is thirteen and goes, and those two differ only in the fraction, so this is not a magnitude threshold. And a column of `[1.234, 123456789.0]` at precision two is short enough after stripping to stay fixed even though it has a value over a million in it, so the magnitude alone does not do it either.

This also means the format is decided by the widest value and paid for by all of them. One value of `1e16` in a column of ordinary numbers sends every other value in that column to scientific notation, which is why `[1e16, 2.0]` prints `1.000000e+16` over `2.000000e+00`.

## 5. A value under the last place goes to an exponent on its own

There is a second route to scientific notation and it needs no length at all. If any value in the column is nonzero and smaller in magnitude than the last place that would be printed, the column goes scientific immediately, before the length is even measured.

At six places the boundary is exactly `1e-6`. A column holding `1e-6` prints `0.000001`, because that value is representable in six places, and a column holding `9.9e-7` prints `9.900000e-07`, because it is not. The reason is obvious once stated: writing `9.9e-7` fixed at six places gives `0.000001`, which is a different number, so pandas refuses to do it. A zero is exempt, because zero is exactly representable and a column of zeros staying fixed is what anybody would want.

This rule is why `1e-5` prints as `0.00001` rather than as an exponent. It is above the boundary, it is not long, and there is nothing over a million in the column, so no rule fires and it stays fixed.

## 6. The elision happens first

A frame taller than the row limit throws away its middle before any of this runs, so a value that will not be printed cannot affect what the printed values look like. A column of twelve rows where row six holds `1e16` prints the other eleven in fixed point, because by the time the column is measured row six is not in it.

That ordering is not a detail of our implementation, it is pandas' ordering and it is observable, and it is the reason the renderer gathers the rows it will show and passes only those to the formatter rather than formatting the column and then cutting it.

## 7. Each column decides alone, and the labels are a column

A frame runs all of this once per column and the columns do not consult each other. A frame with a float column holding `1e16` beside a float column holding `1.0` prints the first in scientific notation and the second as `1.0`, on the same rows of the same output, which looks inconsistent and is exactly what pandas does.

The labels down the side are a column too and go through the same thing. A float index with one enormous label in it prints every label in scientific notation, and a float index of ordinary numbers prints them fixed, and this is the same code rather than a copy of it.

## 8. Rounding goes to the even digit

The writing is C's `printf` underneath in pandas, which rounds a tie to the even digit rather than away from zero. `0.0078125` at six places is `0.007812` and not `0.007813`. `2.5` at no places is `2` and `3.5` at no places is `4`. Those are all exact binary values, so the tie is real rather than an artefact of the decimal representation, and it is worth having tests on because the obvious implementation of rounding gets all four of them wrong.

Everything that is not an exact tie rounds the way anybody would expect, and the tie only arises for values with a short binary fraction, which in practice means halves, quarters, eighths and so on.

## 9. What the exponent looks like

The exponent is always signed and always at least two digits, so it is `e+00` and `e-07` and `e+100`. The mantissa is written at the same precision as everything else, and at precision zero `1/3` prints as `3e-01` rather than as `0`, because the small value rule fires before anything is written and one third is smaller than the last place printed when there are no places.

Rounding the mantissa can carry into the exponent. A value just under a power of ten rounds up to `10.000000` at six places, which is not a mantissa, so the exponent goes up by one and the mantissa is written again.

## 10. Two shortcuts that are equivalences

Two things in the implementation are not rules and are worth naming so that nobody reads them as rules.

A magnitude at or above `1e15` takes the scientific branch without being written fixed first. That is not a threshold pandas has. It is that sixteen integer digits plus a point plus six places plus the leading place is twenty four characters, which is over the twelve character limit, and a value that large is also over a million, so both halves of section 4 are satisfied and the answer cannot come out any other way. The shortcut exists because writing it fixed first would need an integer part that does not fit in an `Int`.

A null cell is left out of the length measurement. pandas includes it, with its leading place, so `<NA>` counts as five characters and `NaN` counts as four. Neither can ever be the longest cell in a column that is over the limit, because the limit at six places is twelve, so leaving them out cannot change the answer. If the precision were ever configurable down to one or two this would need revisiting.

## 11. What is left

The stripping and the switch are per column in pandas and per column here, but pandas decides them inside the same object that holds the column's values, and we decide them in the renderer from a gathered list of floats. That is fine for every case in this document and it will stop being fine if a formatter is ever configurable per column, which pandas allows through `float_format` and we do not.

Nothing here applies to a float32 column, which pandas formats through the same path at the same precision and we currently widen on the way in. Nothing here applies to a decimal or a complex column, neither of which exists in this library. `display.precision` is readable and writable here and `display.float_format`, `display.chop_threshold` and `display.max_colwidth` are not.

The rounding matches C's on every tie that has been tested, but ours rounds the product of a subtraction and a multiply and C rounds the exact binary value, so a value where those two disagree in the last place would print differently. No such value has been found and one may well exist.
