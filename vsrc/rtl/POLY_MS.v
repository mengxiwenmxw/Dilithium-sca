module POLY_MS (
  input  wire [23:0] poly_ms_a,
  input  wire [23:0] poly_ms_b,
  input  wire [23:0] poly_ms_q,
  input  wire [4:0]  poly_q_width,
  output wire [23:0] poly_ms_o
);

  wire [24:0] poly_ms_temp0 = poly_ms_a - poly_ms_b;
  wire [24:0] poly_ms_temp1 = poly_ms_temp0 + poly_ms_q;
  assign poly_ms_o = (poly_ms_temp0[poly_q_width] == 1'b1) ? poly_ms_temp1[23:0] : poly_ms_temp0[23:0];

endmodule
