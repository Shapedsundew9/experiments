# Performance of Weighted Random Sub-Sampling in Postgresql

## Method

1. Create a table with 2 columns and 2<sup>N</sup> rows
   1. Column 'id': Indexed value (random 64 bit integer)
   2. Column 'p': Probablity weight (random double precision [0.0, 1.0))
   3. 0 <= N <= 28
2. Select a uniformly random fraction of rows size 2<sup>-M</sup>
   1. 0 <= M <= 8
   2. M <= N
   3. Some ideas here: <https://stackoverflow.com/questions/5297396/quick-random-row-selection-in-postgres>
3. Select a weighted random sample
   1. Use [Weighted Random Sampling (2005; Efraimidis, Spirakis)](http://utopia.duth.gr/~pefraimi/research/data/2007EncOfAlg.pdf) method to select a single row.
   2. Other ideas here: <https://stackoverflow.com/questions/13040246/select-random-row-from-a-postgresql-table-with-weighted-row-probabilities>
4. Record the time it took to retrieve the row.
