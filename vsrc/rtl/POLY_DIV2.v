module POLY_DIV2(
    input  wire [23:0] poly_div2_din,
    input  wire [23:0] poly_div2_q,
    output wire [23:0] poly_div2_dout
);

    wire [23:0] poly_div2_x         = poly_div2_din;
    wire        poly_div2_q_is_odd  = poly_div2_q[0];
    wire        poly_div2_x_is_odd  = poly_div2_x[0];

    wire [24:0] poly_div2_q_ext     = {1'b0, poly_div2_q};
    wire [24:0] poly_div2_inv2_25   = (poly_div2_q_ext + 25'd1) >> 1;   // = (q+1)/2
    wire [23:0] poly_div2_inv2      = poly_div2_inv2_25[23:0];

    wire [23:0] poly_div2_half_even      = poly_div2_x >> 1;

    wire [23:0] poly_div2_half_odd_qodd  = {1'b0, poly_div2_x[23:1]} + poly_div2_inv2;

    assign poly_div2_dout = poly_div2_q_is_odd
                          ? (poly_div2_x_is_odd ? poly_div2_half_odd_qodd : poly_div2_half_even)
                          : poly_div2_half_even;

endmodule