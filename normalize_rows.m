function [out] = normalize_rows(in)
out = in;
for i = 1:size(out, 1)
    row = out(i, :);
    row_scaled = (row - min(row)) / (max(row) - min(row));
    out(i, :) = row_scaled;
end