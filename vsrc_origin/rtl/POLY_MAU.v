`include "poly_parameters.v"
module POLY_MAU(
  input clk,
  input rst_n,
  input poly_kd_sel,//0:kyber  1:dilithium
  input poly_pwm2_odd_even_sel, // for lyber PWM2
  input [1:0] poly_duv_mode,//00:du=10 01:du=11 10:dv=4 11:dv=5
  input [3:0] poly_alu_mode,
  input [1:0] poly_compress, //00:no compress 01:compress 11:decompress
  input [1:0] poly_decompose,//00:no decompose 01:(q-1)/44  11:(q-1)/16
  (* dont_touch = "true" *)(* keep = "TRUE" *)input [23:0] poly_mau_a,
  (* dont_touch = "true" *)(* keep = "TRUE" *)input [23:0] poly_mau_b,
  (* dont_touch = "true" *)(* keep = "TRUE" *)input [23:0] poly_mau_c,
  (* dont_touch = "true" *)(* keep = "TRUE" *)input [23:0] poly_mau_d,
  (* dont_touch = "true" *)(* keep = "TRUE" *)input [23:0] poly_q,
  (* dont_touch = "true" *)(* keep = "TRUE" *)input [24:0] poly_barret_m,
  (* dont_touch = "true" *)(* keep = "TRUE" *)input [4:0] poly_mm_N,
  (* dont_touch = "true" *)(* keep = "TRUE" *)input  poly_enable,
  output poly_valid,
  (* dont_touch = "true" *)(* keep = "TRUE" *) output reg [23:0] poly_mau_o0,
  (* dont_touch = "true" *)(* keep = "TRUE" *) output reg [23:0] poly_mau_o1

);

  wire poly_mr_alpha_o_sel_dff1;
  wire [23:0] poly_mr_alpha_a_dff1;
  wire [23:0] poly_alu_o1_t;
  wire [23:0] poly_alu_o0_t;
  wire [23:0] poly_alu_c_t;
  wire [23:0] poly_alu_d_t;
  wire [23:0] poly_alu_d_t0;
  wire [23:0] poly_alu_d_dff1;
  reg  [23:0] poly_alu_d_dff2;
  reg  [23:0] poly_alu_d_dff3;
  reg  poly_mr_alpha_o_sel_dff2;
  reg  poly_mr_alpha_o_sel_dff3;
  reg  [9:0] poly_mode;

always @(*) begin
  if(poly_kd_sel) begin
    case(poly_alu_mode)
        `NTT256,`NTT512   : poly_mode = 10'b0111001100;//PWM2 a+bw
        `INTT256,`INTT512 : poly_mode = 10'b1001110011;
        `PWM              : poly_mode = 10'b1100100101;//ab
        `PSUB             : poly_mode = 10'b1001110001;
        `DECPS,`MKHT,`USHT
                          : poly_mode = 10'b1001110001;//(c-d)*a
        `PADD, `P2R       : poly_mode = 10'b1001110111;
        default           : poly_mode = 10'b0111001100;
    endcase
  end else begin
    case(poly_alu_mode) // kyber待修�?
        `NTT256,`NTT512   : poly_mode = 10'b0111001100;//PWM2 a+bw
        `INTT256,`INTT512 : poly_mode = 10'b1001110011;
        `PWM, `DECSS,`CSS
                          : poly_mode = 10'b1100100101;//COMPRESS/DECOMPRESS ab
        `PWM2             : poly_mode = (poly_pwm2_odd_even_sel == 1'b1) ? 10'b1100100000 : 10'b0111001100; // ab-c-d / a+bw
        `PSUB             : poly_mode = 10'b1001110001;
        `PADD             : poly_mode = 10'b1001110111;
        default           : poly_mode = 10'b0111001100;
    endcase
  end
end

(* dont_touch = "true" *)(* keep = "TRUE" *) POLY_MR_ALPHA u_MR_ALPHA(
    .clk                 ( clk                 ),
    .rst_n               ( rst_n               ),
    .decompose           ( poly_decompose           ),
    .mr_alpha_a          ( poly_mau_c               ),
    .mr_alpha_o_sel_dff1 ( poly_mr_alpha_o_sel_dff1 ),
    .mr_alpha_a_dff1     ( poly_mr_alpha_a_dff1     ),
    .mr_alpha_o          ( poly_alu_d_t             ),
    .mr_alpha_o_dff1     ( poly_alu_d_dff1          )
);


(* dont_touch = "true" *)(* keep = "TRUE" *) POLY_ALU u_POLY_ALU(
  .poly_clk       ( clk       ),
  .poly_rst_n     ( rst_n     ),
  .poly_enable    ( poly_enable    ),
  .poly_decompose ( poly_decompose ),
  .poly_duv_mode  ( poly_duv_mode  ),
  .poly_compress  ( poly_compress  ),
  .poly_mode      ( poly_mode      ),
  .poly_data_in0  ( poly_mau_a  ),
  .poly_data_in1  ( poly_mau_b  ),
  .poly_data_in2  ( poly_alu_c_t  ),
  .poly_data_in3  ( poly_alu_d_t0  ),
  .poly_q         ( poly_q         ),
  .poly_barret_m  ( poly_barret_m  ),
  .poly_mm_N      ( poly_mm_N      ),
  .poly_valid     ( poly_valid     ),
  .poly_data_out0 ( poly_alu_o0_t ),
  .poly_data_out1  ( poly_alu_o1_t  )
);


assign poly_alu_d_t0 = poly_decompose[0] ? poly_alu_d_t : poly_mau_d ;
assign poly_alu_c_t  = poly_decompose[0] ? poly_mr_alpha_a_dff1 : poly_mau_c ;

always @(posedge clk or negedge rst_n) begin
  if(~rst_n)
  begin
    poly_mau_o0 <= 24'b0;
    poly_mau_o1 <= 24'b0;
    poly_mr_alpha_o_sel_dff2 <= 1'b0;
    poly_mr_alpha_o_sel_dff3 <= 1'b0;
    poly_alu_d_dff2 <= 24'b0;
    poly_alu_d_dff3 <= 24'b0;
  end
  else begin
    poly_alu_d_dff2 <= poly_alu_d_dff1;
    poly_alu_d_dff3 <= poly_alu_d_dff2;
    poly_mr_alpha_o_sel_dff2 <= poly_mr_alpha_o_sel_dff1;
    poly_mr_alpha_o_sel_dff3 <= poly_mr_alpha_o_sel_dff2;
    poly_mau_o0 <= poly_decompose[0] ? poly_alu_d_dff1 : poly_alu_o0_t;
    poly_mau_o1 <= poly_decompose[0] ? (poly_mr_alpha_o_sel_dff1 ? 24'b0 : poly_alu_o1_t) : poly_alu_o1_t;
  end
end

endmodule
