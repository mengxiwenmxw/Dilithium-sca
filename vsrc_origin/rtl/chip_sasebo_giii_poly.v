/*-------------------------------------------------------------------------
 AES cryptographic module for FPGA on SASEBO-GIII
 
 File name   : chip_sasebo_giii_aes.v
 Version     : 1.0
 Created     : APR/02/2012
 Last update : APR/25/2013
 Desgined by : Toshihiro Katashita
 
 
 Copyright (C) 2012,2013 AIST
 
 By using this code, you agree to the following terms and conditions.
 
 This code is copyrighted by AIST ("us").
 
 Permission is hereby granted to copy, reproduce, redistribute or
 otherwise use this code as long as: there is no monetary profit gained
 specifically from the use or reproduction of this code, it is not sold,
 rented, traded or otherwise marketed, and this copyright notice is
 included prominently in any copy made.
 
 We shall not be liable for any damages, including without limitation
 direct, indirect, incidental, special or consequential damages arising
 from the use of this code.
 
 When you publish any results arising from the use of this code, we will
 appreciate it if you can cite our paper.
 (http://www.risec.aist.go.jp/project/sasebo/)
 -------------------------------------------------------------------------*/


//================================================ CHIP_SASEBO_GIII_AES
module CHIP_SASEBO_GIII_POLY
  (// Local bus for GII
   lbus_di_a, //地址和数�????????
   lbus_do,   //
   lbus_wrn, lbus_rdn,
   lbus_clkn, lbus_rstn,

   // GPIO and LED
   gpio_startn, 
   gpio_endn, gpio_exec, led,   //这三个不用管

   // Clock OSC
   osc_en_b //晶振
   );
   
   //------------------------------------------------
   // Local bus for GII
   (* keep = "TRUE" *)input [15:0]  lbus_di_a;
   (* keep = "TRUE" *)output [15:0] lbus_do;
   (* keep = "TRUE" *)input         lbus_wrn, lbus_rdn;
   (* keep = "TRUE" *)input         lbus_clkn, lbus_rstn;

   // GPIO and LED
   (* keep = "TRUE" *)output        gpio_startn, gpio_endn, gpio_exec;
   (* keep = "TRUE" *)output [9:0]  led;

   // Clock OSC
   (* keep = "TRUE" *)output        osc_en_b;

   //------------------------------------------------
   // Internal clock
   (* keep = "TRUE" *)wire         clk, rst;

   // Local bus
   (* keep = "TRUE" *)reg [15:0]   lbus_a, lbus_di;
   
   // Block cipher
   (* keep = "TRUE" *)wire [127:0] blk_kin, blk_din, blk_dout;
   (* keep = "TRUE" *)wire         blk_krdy, blk_kvld, blk_drdy, blk_dvld;
   (* keep = "TRUE" *)wire         blk_encdec, blk_en, blk_rstn, blk_busy;
   (* keep = "TRUE" *)reg          blk_drdy_delay;
  
   //------------------------------------------------
   assign led[0] = rst;
   assign led[1] = lbus_rstn;
   assign led[2] = 1'b0;
   assign led[3] = blk_rstn;
   assign led[4] = blk_encdec;
   assign led[5] = blk_krdy;
   assign led[6] = blk_kvld;
   assign led[7] = 1'b0;
   assign led[8] = blk_dvld;
   assign led[9] = blk_busy;

   assign osc_en_b = 1'b0;
   //------------------------------------------------
   always @(posedge clk) if (lbus_wrn)  lbus_a  <= lbus_di_a;  //1 写地�??
   always @(posedge clk) if (~lbus_wrn) lbus_di <= lbus_di_a;  //0 写数�??

   //(* keep = "TRUE" *)wire[11:0] a,b;
   (* keep = "TRUE" *) wire[23:0]a;
   (* keep = "TRUE" *) wire[23:0]b;
  

   LBUS_IF lbus_if
     (.lbus_a(lbus_a), .lbus_di(lbus_di), .lbus_do(lbus_do),
      .lbus_wr(lbus_wrn), .lbus_rd(lbus_rdn),
      .a(a),.b(b),
      .blk_dout(blk_dout),
      .blk_krdy(blk_krdy), .blk_drdy(blk_drdy), 
      .blk_kvld(blk_kvld), .blk_dvld(blk_dvld),
      .blk_en(blk_en), .blk_rstn(blk_rstn),
      .clk(clk), .rst(rst));
   

   //------------------------------------------------
   assign gpio_startn = ~blk_drdy;
   assign gpio_endn   = 1'b0; //~blk_dvld;
   assign gpio_exec   = 1'b0; //blk_busy;

   always @(posedge clk) blk_drdy_delay <= blk_drdy;
   wire rst_n = ~rst;

   //------------------------------------------------   
   MK_CLKRST mk_clkrst (.clkin(lbus_clkn), .rstnin(lbus_rstn),
                        .clk(clk), .rst(rst));

// ******************** User add code ********************
// generate working_flag

   `define WORKING_CYCLE_NUM 3'd6
   
   (* keep = "TRUE" *) reg working_flag;
   (* keep = "TRUE" *) reg [2:0]working_cycle_cnt;
   (* keep = "TRUE" *) wire start;

   assign start = ~gpio_startn;
   
   always @(posedge clk or negedge rst_n) begin
      if (~rst_n) begin
         working_flag <= 1'b0;
         working_cycle_cnt <= 3'd0;
      end else begin
         if(start)begin
            working_flag <= 1'b1;
            working_cycle_cnt <= 3'b0;
         end else if (working_flag) begin 
            if(working_cycle_cnt<`WORKING_CYCLE_NUM-1'b1) begin
               working_cycle_cnt <= working_cycle_cnt + 1'b1;
            end else begin
               working_flag <= 1'b0;
               working_cycle_cnt <= 3'd0;
            end
         end
      end
   end

   //genvar i;
   //generate
      //for(i= 0; i<1; i=i+1) begin
      
        `define PWM     4'b0100
        `define qNUM    24'd3329
        `define mNUM    25'd5039
        `define nNUM    5'd12
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
assign poly_enable = working_flag;

genvar i;

generate
    for(i=0;i<10;i=i+1 )begin
(* dont_touch = "true"*) (* keep = "true" *) POLY_MAU u_POLY_MAU(
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
     end
   endgenerate
 

// ******************** User add code finish ********************
   
endmodule 


   
//================================================ MK_CLKRST
module MK_CLKRST (clkin, rstnin, clk, rst);
   //synthesis attribute keep_hierarchy of MK_CLKRST is no;
   
   //------------------------------------------------
   input  clkin, rstnin;
   output clk, rst;
   
   //------------------------------------------------
   wire   refclk;
//   wire   clk_dcm, locked;

   //------------------------------------------------ clock
   IBUFG u10 (.I(clkin), .O(refclk)); 

/*
   DCM_BASE u11 (.CLKIN(refclk), .CLKFB(clk), .RST(~rstnin),
                 .CLK0(clk_dcm),     .CLKDV(),
                 .CLK90(), .CLK180(), .CLK270(),
                 .CLK2X(), .CLK2X180(), .CLKFX(), .CLKFX180(),
                 .LOCKED(locked));
   BUFG  u12 (.I(clk_dcm),   .O(clk));
*/

   BUFG  u12 (.I(refclk),   .O(clk));
   // changed
   // assign clk = ~clkin;

   //------------------------------------------------ reset
   MK_RST u20 (.locked(rstnin), .clk(clk), .rst(rst));
endmodule // MK_CLKRST



//================================================ MK_RST
module MK_RST (locked, clk, rst);
   //synthesis attribute keep_hierarchy of MK_RST is no;
   
   //------------------------------------------------
   input  locked, clk;
   output rst;

   //------------------------------------------------
   // delay for fpga
   reg [15:0] cnt;
   
   //------------------------------------------------
   always @(posedge clk or negedge locked) 
     if (~locked)    cnt <= 16'h0;
     else if (~&cnt) cnt <= cnt + 16'h1;

   // changed in 251102-mxw
   assign rst = ~&cnt;
   // assign rst = ~locked;
endmodule // MK_RST
