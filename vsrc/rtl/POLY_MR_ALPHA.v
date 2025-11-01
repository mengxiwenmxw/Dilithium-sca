`include "poly_parameters.v"

module POLY_MR_ALPHA (
  input clk,
  input rst_n,
  input [1:0] decompose, //0: alpha = (q-1)/44=18'd190464  1: alpha = (q-1)/16=19'523776
  input [23:0] mr_alpha_a,
  output reg mr_alpha_o_sel_dff1,//0: r-r0 != q-1  1: r-r0 == q-1
  output reg [23:0] mr_alpha_a_dff1,
  output  [23:0] mr_alpha_o,
  output reg [23:0] mr_alpha_o_dff1
);

wire [3:0] mr_alpha_a_h0;
reg [18:0] mr_alpha_lut_h0;
wire [2:0] mr_alpha_a_h10;
wire [1:0] mr_alpha_a_h11;
reg [17:0] mr_alpha_lut_h10;
reg [18:0] mr_alpha_lut_h11;
wire [17:0] mr_alpha_lut_h1;

wire [18:0] mr_alpha_lut_h;
wire [18:0] mr_alpha_lut_l;

wire [18:0] mr_alpha_l;
wire [19:0] mr_alpha_l_sub;

reg [19:0] mr_alpha_add;
reg [19:0] mr_alpha_sub;
reg [23:0] mr_alpha_a_dff0;
reg [23:0] mr_alpha_o_dff0;
reg mr_alpha_o_sel;
reg mr_alpha_o_sel_dff0;


assign mr_alpha_a_h0  = mr_alpha_a[22:19];
assign mr_alpha_a_h10 = mr_alpha_a[20:18];
assign mr_alpha_a_h11 = mr_alpha_a[22:21];
assign mr_alpha_lut_h1 = (mr_alpha_lut_h11 + mr_alpha_lut_h10) >= `ALPHA0 ? mr_alpha_lut_h11 + mr_alpha_lut_h10 - `ALPHA0 : mr_alpha_lut_h11 + mr_alpha_lut_h10;
assign mr_alpha_lut_h = decompose == 2'b11 ? mr_alpha_lut_h0 : {1'b0, mr_alpha_lut_h1};

assign mr_alpha_l = decompose == 2'b11 ? mr_alpha_a[18:0] : {1'b0, mr_alpha_a[17:0]};
assign mr_alpha_l_sub = decompose == 2'b11 ? mr_alpha_l - `ALPHA1 : mr_alpha_l - `ALPHA0;
assign mr_alpha_lut_l = decompose == 2'b11 ? (mr_alpha_l_sub[19] ? mr_alpha_l : mr_alpha_l_sub) : (mr_alpha_l_sub[18] ? mr_alpha_l : mr_alpha_l_sub); 

always @(posedge clk or negedge rst_n) begin
  if(~rst_n) mr_alpha_add <= 20'd0;
  else mr_alpha_add <= mr_alpha_lut_h + mr_alpha_lut_l;
end

always @(posedge clk or negedge rst_n) begin
  if(~rst_n) mr_alpha_sub <= 20'd0;
  else begin
    if(decompose == 2'b11)
    begin
      if(mr_alpha_add <= `ALPHA1/2)  mr_alpha_sub <= mr_alpha_add[19:0];
      else if(mr_alpha_add <= 3*`ALPHA1/2) mr_alpha_sub <= mr_alpha_add - `ALPHA1;
      else  mr_alpha_sub <= mr_alpha_add - 2*`ALPHA1;
    end
    else begin
      if(mr_alpha_add <= `ALPHA0/2)  mr_alpha_sub <= mr_alpha_add[18:0];
      else if(mr_alpha_add <= 3*`ALPHA0/2) mr_alpha_sub <= mr_alpha_add - `ALPHA0;
      else  mr_alpha_sub <= mr_alpha_add - 2*`ALPHA0;
    end
  end
end

always @(posedge clk or negedge rst_n) begin
  if(~rst_n)
  begin
    mr_alpha_a_dff1 <= 24'd0;
    mr_alpha_a_dff0 <= 24'd0;
    mr_alpha_o_sel_dff1 <= 1'b0;
    mr_alpha_o_sel_dff0 <= 1'b0;
    mr_alpha_o_dff1 <= 1'b0;
    mr_alpha_o_dff0 <= 1'b0;
  end
  else begin
    mr_alpha_a_dff1 <= mr_alpha_a_dff0;
    mr_alpha_a_dff0 <= mr_alpha_a;
    mr_alpha_o_sel_dff1 <= mr_alpha_o_sel_dff0;
    mr_alpha_o_sel_dff0 <= mr_alpha_o_sel;
    mr_alpha_o_dff1 <= mr_alpha_o_dff0;
    mr_alpha_o_dff0 <= mr_alpha_o;
  end
end

always @(posedge clk or negedge rst_n) begin
  if(~rst_n) mr_alpha_o_sel <= 1'd0;
  else mr_alpha_o_sel <= decompose == 2'b11 ? mr_alpha_a_dff0 >= 23'd8118529 :  mr_alpha_a_dff0 >= 23'd8285185 ;
end

assign mr_alpha_o = mr_alpha_o_sel ? $signed(mr_alpha_sub - 1'b1) : $signed(mr_alpha_sub);

always@(*) //ROM_H
begin
	case(mr_alpha_a_h0)
    4'd0:  mr_alpha_lut_h0 = 19'd0;
    4'd1:  mr_alpha_lut_h0 = 19'd512;
    4'd2:  mr_alpha_lut_h0 = 19'd1024;
    4'd3:  mr_alpha_lut_h0 = 19'd1536;
    4'd4:  mr_alpha_lut_h0 = 19'd2048;
    4'd5:  mr_alpha_lut_h0 = 19'd2560;
    4'd6:  mr_alpha_lut_h0 = 19'd3072;
    4'd7:  mr_alpha_lut_h0 = 19'd3584;
    4'd8:  mr_alpha_lut_h0 = 19'd4096;
    4'd9:  mr_alpha_lut_h0 = 19'd4608;
    4'd10: mr_alpha_lut_h0 = 19'd5120;
    4'd11: mr_alpha_lut_h0 = 19'd5632;
    4'd12: mr_alpha_lut_h0 = 19'd6144;
    4'd13: mr_alpha_lut_h0 = 19'd6656;
    4'd14: mr_alpha_lut_h0 = 19'd7168;
    4'd15: mr_alpha_lut_h0 = 19'd7680;
	endcase
end

always@(*) //ROM_H
begin
	case(mr_alpha_a_h10)
    3'd0:  mr_alpha_lut_h10 = 18'd0;
    3'd1:  mr_alpha_lut_h10 = 18'd71680;
    3'd2:  mr_alpha_lut_h10 = 18'd143360;
    3'd3:  mr_alpha_lut_h10 = 18'd24576;
    3'd4:  mr_alpha_lut_h10 = 18'd96256;
    3'd5:  mr_alpha_lut_h10 = 18'd167936;
    3'd6:  mr_alpha_lut_h10 = 18'd49152;
    3'd7:  mr_alpha_lut_h10 = 18'd120832;
	endcase

	case(mr_alpha_a_h11)
    2'd0:  mr_alpha_lut_h11 = 18'd0;
    2'd1:  mr_alpha_lut_h11 = 18'd2048;
    2'd2:  mr_alpha_lut_h11 = 18'd4096;
    2'd3:  mr_alpha_lut_h11 = 18'd6144;
	endcase  
end
  
endmodule
