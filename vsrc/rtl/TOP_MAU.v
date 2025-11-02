`define PWM     4'b0100
`define qNUM    24'd8380417
`define mNUM    25'd8396807
`define nNUM    5'd23


module TOP_MAU(clk,rst_n,a,b,enable);
input clk;
input rst_n;
input [23:0]a;
input [23:0]b;
input enable;

wire poly_kd_sel;//0:kyber  1:dilithium
wire poly_pwm2_odd_even_sel; // for lyber PWM2
wire [1:0] poly_duv_mode;//00:du=10 01:du=11 10:dv=4 11:dv=5
wire [3:0] poly_alu_mode;
wire [1:0] poly_compress; //00:no compress 01:compress 11:decompress
wire [1:0] poly_decompose;//00:no decompose 01:(q-1)/44  11:(q-1)/16

wire [23:0] poly_mau_a; // mm num1
wire [23:0] poly_mau_b; // mm num2
wire poly_enable;

wire [23:0] poly_mau_c; // 0
wire [23:0] poly_mau_d; // 0
wire [23:0] poly_q;
wire [24:0] poly_barret_m;
wire [4:0] poly_mm_N;


wire poly_valid;
wire [23:0] poly_mau_o0;
wire [23:0] poly_mau_o1;

assign poly_kd_sel = 1'b1;//Dilithium
assign poly_pwm2_odd_even_sel = 1'b0;
assign poly_duv_mode = 2'b00;
assign poly_alu_mode = `PWM;
assign poly_compress = 2'b00;
assign poly_decompose = 2'b00;

assign poly_mau_c = 24'd0;
assign poly_mau_d = 24'd0;
assign poly_q = `qNUM;
assign poly_barret_m = `mNUM;
assign poly_mm_N = `nNUM;

// Ctr Logic
assign poly_mau_a = a;
assign poly_mau_b = b;
assign poly_enable = enable;



POLY_MAU u_POLY_MAU(
  .clk(clk),
  .rst_n(rst_n),
  .poly_kd_sel(poly_kd_sel),//0:kyber  1:dilithium
  .poly_pwm2_odd_even_sel(poly_pwm2_odd_even_sel), // for lyber PWM2
  .poly_duv_mode(poly_duv_mode),//00:du=10 01:du=11 10:dv=4 11:dv=5
  .poly_alu_mode(poly_alu_mode),
  .poly_compress(poly_compress), //00:no compress 01:compress 11:decompress
  .poly_decompose(poly_decompose),//00:no decompose 01:(q-1)/44  11:(q-1)/16
  .poly_mau_a(poly_mau_a),
  .poly_mau_b(poly_mau_b),
  .poly_mau_c(poly_mau_c),
  .poly_mau_d(poly_mau_d),
  .poly_q(poly_q),
  .poly_barret_m(poly_barret_m),
  .poly_mm_N(poly_mm_N),
  .poly_enable(poly_enable),
  .poly_valid(poly_valid),
  .poly_mau_o0(poly_mau_o0),
  .poly_mau_o1(poly_mau_o1)

);

endmodule