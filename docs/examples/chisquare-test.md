# Chi‑Square Goodness‑of‑Fit Test for Continuous Data

When working with **continuous data** and you want to test whether the data come from a specific probability distribution,
a common approach is to bin the data and apply **Pearson’s chi‑square goodness‑of‑fit test**.  This test compares the
observed frequencies in each bin with the frequencies expected under the hypothesised distribution.

## Definition of the chi‑square statistic

If $O_i$ is the observed number of data points in the $i$-th bin and $E_i$ is the expected number under the null
hypothesis, then the chi‑square statistic is

$$
\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}.
$$

Under the null hypothesis, $\chi^2$ approximately follows a chi‑square distribution with $k-1$ degrees of freedom.  If you estimate $p$ parameters from the data, you must reduce the degrees of freedom to $k-1-p$.

## Applying the chi‑square test to continuous data

1. **Select a binning scheme.**  Divide the range of your data into $k$ bins using rules such as Sturges’ rule $k = \lceil \log_2(n)+1 \rceil$.  Ensure each bin is wide enough that its expected frequency is not too small.
2. **Count observations.**  Count the number of data points in each bin; these are the observed counts $O_i$.
3. **Compute expected counts.**  For a hypothesised distribution with cumulative distribution function $F(x)$, the probability that a value falls in bin $[a_i, a_{i+1})$ is $F(a_{i+1}) - F(a_i)$.  Multiply by the total number of data points to get $E_i$.  For example, the Gumbel distribution with location $\mu$ and scale $\beta$ has $F(x;\mu,\beta) = \exp\bigl(-\exp[-(x-\mu)/\beta]\bigr)$.
4. **Merge bins with small expected counts.**  Bins with expected counts less than 5 should be merged with adjacent bins.  After merging, recompute $k$.
5. **Match totals.**  Rescale the expected counts if necessary so that $\sum_i E_i = \sum_i O_i$.
6. **Compute $\chi^2$ and the p‑value.**  Calculate the statistic above and obtain the p‑value from the chi‑square distribution with $k-1-p$ degrees of freedom.  A small p‑value (e.g., below 0.05) suggests that the data do not follow the hypothesised distribution.

## Example: Testing a Gumbel fit

Assume you have 1000 observations that have been fitted to a Gumbel distribution with parameters $\mu = 0.0101$ and $\beta = 1.0313$.  Using Sturges’ rule yields 11 bins.  The initial observed counts and expected counts are shown in the first figure below; bins with expected counts below 5 are highlighted for merging.

![Observed vs expected counts using Sturges bins]({{file:file-KdALjVHE8cimJCRxYJ19Co}})

After merging the rightmost bins to ensure each expected count is at least 5, there are eight bins.  Since two parameters were estimated ($\mu$ and $\beta$), the degrees of freedom for the chi‑square test are $8-1-2 = 5$.

![Merged bins and expected counts]({{file:file-2pmFtBMVX4Ss4qbJxM734b}})

The contributions of each bin to $\chi^2$ are illustrated in the third figure.  Summing these contributions yields $\chi^2 \approx 2.69$.  With 5 degrees of freedom, the p‑value is about 0.75, which is not significant; hence the data are consistent with the fitted Gumbel distribution.

![Chi‑square contributions per merged bin]({{file:file-3tVWFqUiT3ht9LrXrQQTRk}})

## Notes on citations and formatting

The reference labels such as correspond to the specific sources used in this document.  If you integrate this file into a documentation site (e.g., MkDocs) that does not support these citation tags, you may choose to remove them.  Likewise, ensure your Markdown parser supports LaTeX syntax (e.g., enable `markdown_katex` or `pymdownx.arithmatex`) so that equations enclosed in `$$…$$` render correctly.
