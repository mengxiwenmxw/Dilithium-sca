module POLY_MA (
  input  wire [23:0] poly_ma_a,
  input  wire [23:0] poly_ma_b,
  input  wire [23:0] poly_ma_q,
  output wire [23:0] poly_ma_o
);

  wire [24:0] poly_ma_sum   = poly_ma_a + poly_ma_b;
  assign poly_ma_o = (poly_ma_sum < poly_ma_q) ? poly_ma_sum[23:0] : (poly_ma_sum - poly_ma_q);

endmodule
