`timescale 1ns/100ps

`define PWM     4'b0100
`define qNUM    24'd3329  // -> 3329
`define mNUM    25'd5039  // -> 2^24 / 3329
`define nNUM    5'd12        // -> 12

`define FINISH_TIME  8400000

module tb_POLY_MAU;

reg clk;
reg rst_n;
wire poly_kd_sel;//0:kyber  1:dilithium
reg poly_pwm2_odd_even_sel; // for lyber PWM2
wire [1:0] poly_duv_mode;//00:du=10 01:du=11 10:dv=4 11:dv=5
wire [3:0] poly_alu_mode;
wire [1:0] poly_compress; //00:no compress 01:compress 11:decompress
wire [1:0] poly_decompose;//00:no decompose 01:(q-1)/44  11:(q-1)/16
reg [23:0] poly_mau_a; // mm num1
reg [23:0] poly_mau_b; // mm num2
wire [23:0] poly_mau_c; // 0
wire [23:0] poly_mau_d; // 0
wire [23:0] poly_q;
wire [24:0] poly_barret_m;
wire [4:0] poly_mm_N;
reg  poly_enable;

wire poly_valid;
wire [23:0] poly_mau_o0;
wire [23:0] poly_mau_o1;

assign poly_kd_sel = 1'b1;//Dilithium -> 3bits
assign poly_duv_mode = 2'b00;
assign poly_alu_mode = `PWM;  //
assign poly_compress = 2'b00;
assign poly_decompose = 2'b00;

assign poly_mau_c = 24'd0;
assign poly_mau_d = 24'd0;
assign poly_q = `qNUM;
assign poly_barret_m = `mNUM;
assign poly_mm_N = `nNUM;

parameter CLK_PERIOD = 10;
// *************** INIT *************** 




// initial begin
//     $fsdbDumpfile("waves/wave.fsdb");
//     $fsdbDumpvars(0, tb_POLY_MAU);
//     rst_n = 0 ;
//     poly_enable = 0;
//     #10 rst_n = 1 ;
//     poly_enable = 1 ;
//     #`FINISH_TIME;
//     $finish;

// end

// always @(posedge clk or negedge rst_n)begin
// if(~rst_n)begin
//   poly_mau_a <= 24'd0;
//   poly_mau_b <= 24'd2773;
// end
// else begin
//   if(poly_enable)begin
//     if(poly_mau_a<poly_q)begin
//       poly_mau_a <= poly_mau_a + 1'b1;
//     end else begin
//       poly_mau_a <= 24'd0;
//       poly_mau_b <= poly_mau_b + 1'b1;
//     end
//   end
// end
// end
parameter DATA_WIDTH = 24;
parameter NUM_INPUTS = 6;

task pipeline_input;
  input [DATA_WIDTH-1:0] data_a [NUM_INPUTS-1:0];
  integer i;
  begin
    for (i = 0; i < NUM_INPUTS; i = i + 1) begin
        @(posedge clk);
        poly_enable = 1;
        poly_mau_a <= data_a[i];
      end
    @(posedge clk);
    poly_enable = 0;
    poly_mau_a <= 0;
  end
endtask

always #((CLK_PERIOD)/2) clk = ~clk; // 100MHz 时钟

reg [DATA_WIDTH-1:0] test_data [NUM_INPUTS-1:0];

initial begin
  $fsdbDumpfile("waves/tb_POLY_MAU.fsdb");
  $fsdbDumpvars(0, tb_POLY_MAU);
  clk = 0;
  poly_enable = 0;
  #(CLK_PERIOD * 10);
  
  poly_mau_b = 24'd2773;
  #(CLK_PERIOD * 5);

  test_data[0] = 24'd245;
  test_data[1] = 24'd1603;
  test_data[2] = 24'd497;
  test_data[3] = 24'd940;
  test_data[4] = 24'd2815;
  test_data[5] = 24'd2961;

  pipeline_input(test_data);

  #(CLK_PERIOD * 20);
  $finish;
end


POLY_MAU u_POLY_MAU(
  .clk(clk),
  .rst_n(rst_n),
  .poly_kd_sel(poly_kd_sel),//0:kyber  1:dilithium
  .poly_pwm2_odd_even_sel(poly_pwm2_odd_even_sel), // for kyber PWM2
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