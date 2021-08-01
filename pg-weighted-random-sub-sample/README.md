# Performance of Weight Random Sub-Sampling in Postgresql

## Method

1. Create a table with 2 columns and 2<sup>N</sup> rows
   1. Column1: Indexed value (random 64 bit integer)
   2. Column2: Probablity weight (random double precision [0, 1])
   3. 0 <= N <= 28
2. Select a random fraction of rows size 2<sup>-M</sup>
   1. 0 <= M <= 8
   2. M <= N
   3. Some ideas here: <https://stackoverflow.com/questions/5297396/quick-random-row-selection-in-postgres>
3. Use [Weighted Random Sampling (2005; Efraimidis, Spirakis)](http://utopia.duth.gr/~pefraimi/research/data/2007EncOfAlg.pdf) method to select a single row.
   1. Other ideas here: <https://stackoverflow.com/questions/13040246/select-random-row-from-a-postgresql-table-with-weighted-row-probabilities>
4. Record the time it took to retrieve the row.