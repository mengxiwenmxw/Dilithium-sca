/*-------------------------------------------------------------------------
 AIST-LSI compatible local bus I/F for AES_Comp on FPGA
 *** NOTE *** 
 This circuit works only with AES_Comp.
 Compatibility for another cipher module may be provided in future release.
 
 File name   : lbus_if.v
 Version     : 1.3
 Created     : APR/02/2012
 Last update : APR/11/2012
 Desgined by : Toshihiro Katashita
 
 
 Copyright (C) 2012 AIST
 
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
 appreciate it if you can cite our webpage.
(http://www.risec.aist.go.jp/project/sasebo/)
 -------------------------------------------------------------------------*/ 


//================================================ LBUS_IF
module LBUS_IF(
  // input signals
  input [15:0]        lbus_a , // Address
  input [15:0]        lbus_di, // Input data  (Controller -> Cryptographic module)
  input               lbus_wr, // Assert input data
  input               lbus_rd, // Assert output data
  output reg [15:0]   lbus_do, // Output data (Cryptographic module -> Controller)
  // to hardware 
  output reg [23:0]   a ,
  output reg [23:0]   b ,
  output reg          blk_krdy, 
  output              blk_drdy, // blk_drdy assigned to gpio
  output              blk_en  , 
  output reg          blk_rstn,
  // from hardware
  input [127:0]       blk_dout,
  input               blk_kvld, 
  input               blk_dvld,

  input               clk, 
  input               rst       // high valid
);   

  // Block cipher
  // (* dont_touch = "true" *)(* keep = "TRUE" *) output reg [23:0] a;
  // (* dont_touch = "true" *)(* keep = "TRUE" *) output reg [23:0] b;
    (* keep = "true" *) reg [23:0] fifo_a [5:0]; 

   //------------------------------------------------

   reg [256:0]   Key1,  Key2;
   reg [3:0]     Wots_Tree_index;
   reg [6:0]     Wots_Tree_height;
   reg [3:0]     Wots_KeyPair;
   reg [63:0]    Sel_Tree;
   reg [4:0]     Wots_Layer_Addr;

   reg [127:0] 	 blk_dout_reg;
   reg[1:0]      Wots_Mode;
   // wire          blk_en = 1;
   assign blk_en = 1'b1;
   
   reg [1:0]     wr;
   reg           trig_wr;
   wire          ctrl_wr;
   reg [2:0]     ctrl;
   reg [3:0]     blk_trig;

   //------------------------------------------------
   always @(posedge clk or posedge rst)
     if (rst) wr <= 2'b00;
     else     wr <= {wr[0],lbus_wr};
   
   always @(posedge clk or posedge rst)
     if (rst)            trig_wr <= 0;
     else if (wr==2'b01) trig_wr <= 1;
     else                trig_wr <= 0;
   
   assign ctrl_wr = (trig_wr & (lbus_a==16'h0002));
   
   always @(posedge clk or posedge rst) 
     if (rst) ctrl <= 3'b000;
     else begin
        if (blk_drdy)       ctrl[0] <= 1;
        else if (|blk_trig) ctrl[0] <= 1;
        else if (blk_dvld)  ctrl[0] <= 0;

        if (blk_krdy)      ctrl[1] <= 1;
        else if (blk_kvld) ctrl[1] <= 0;
        
        ctrl[2] <= ~blk_rstn;
     end

   always @(posedge clk or posedge rst) 
     if (rst)           blk_dout_reg <= 128'h0;
     else if (blk_dvld) blk_dout_reg <= blk_dout;
   
   always @(posedge clk or posedge rst) 
     if (rst)          blk_trig <= 4'h0;
     else if (ctrl_wr) blk_trig <= {lbus_di[0],3'h0};
     else              blk_trig <= {1'h0,blk_trig[3:1]};
   assign blk_drdy = blk_trig[0];

   always @(posedge clk or posedge rst) 
     if (rst)          blk_krdy <= 0;
     else if (ctrl_wr) blk_krdy <= lbus_di[1];
     else              blk_krdy <= 0;

   always @(posedge clk or posedge rst) 
     if (rst)          blk_rstn <= 1;
     else if (ctrl_wr) blk_rstn <= ~lbus_di[2];
     else              blk_rstn <= 1;
   
   //------------------------------------------------
   reg [2:0] a_out_cnt;
   reg       is_sending;

   always @(posedge clk or posedge rst) begin
       if (rst) begin
           a <= 24'd0;
           a_out_cnt <= 3'd0;
           is_sending <= 1'b0;
       end else begin
           if (is_sending) begin
               a <= fifo_a[a_out_cnt]; 
               if (a_out_cnt == 3'd5) begin
                   is_sending <= 1'b0;
                   a_out_cnt <= 3'd0;
               end else begin
                   a_out_cnt <= a_out_cnt + 1;
               end
           end else if (blk_drdy) begin
               is_sending <= 1'b1;
               a <= fifo_a[0];
               a_out_cnt <= 3'd1;
           end else begin
               a <= 24'd0; 
           end
       end
   end

   // Receive a b
   always @(posedge clk or posedge rst) begin
      if (rst) begin
          b <= 24'd0;
      end else if (trig_wr) begin
        case(lbus_a)
        // a0
         16'h0100 : fifo_a[0] <= lbus_di;
         16'h0101 : fifo_a[0] <={lbus_di,fifo_a[0][15:0]} ;
        // a1
         16'h0102 : fifo_a[1] <= lbus_di;
         16'h0103 : fifo_a[1] <= {lbus_di, fifo_a[1][15:0]};
        // a2
         16'h0104 : fifo_a[2] <= lbus_di;
         16'h0105 : fifo_a[2] <= {lbus_di, fifo_a[2][15:0]};
        // a3
         16'h0106 : fifo_a[3] <= lbus_di;
         16'h0107 : fifo_a[3] <= {lbus_di, fifo_a[3][15:0]};
        // a4
         16'h0108 : fifo_a[4] <= lbus_di;
         16'h0109 : fifo_a[4] <= {lbus_di, fifo_a[4][15:0]};
        // a5
         16'h010a : fifo_a[5] <= lbus_di;
         16'h010b : fifo_a[5] <= {lbus_di, fifo_a[5][15:0]};
        // b
         16'h0110 : b <= lbus_di;
         16'h0111 : b <= {lbus_di,b[15:0]};
        endcase
      end
   end
                
   //------------------------------------------------
   always @(posedge clk or posedge rst)
     if (rst) 
       lbus_do <= 16'h0;
     else if (~lbus_rd)
       lbus_do <= mux_lbus_do(lbus_a, ctrl, Wots_Mode, blk_dout);
   
   function  [15:0] mux_lbus_do;
      input [15:0]   lbus_a;
      input [2:0]    ctrl;
      input [1:0]        Wots_Mode;
      input [127:0]  blk_dout;
      
      case(lbus_a)
        16'h0002: mux_lbus_do = ctrl;
        16'h000C: mux_lbus_do = Wots_Mode;
        16'h0180: mux_lbus_do = blk_dout_reg[127:112];
        16'h0182: mux_lbus_do = blk_dout_reg[111:96];
        16'h0184: mux_lbus_do = blk_dout_reg[95:80];
        16'h0186: mux_lbus_do = blk_dout_reg[79:64];
        16'h0188: mux_lbus_do = blk_dout_reg[63:48];
        16'h018A: mux_lbus_do = blk_dout_reg[47:32];
        16'h018C: mux_lbus_do = blk_dout_reg[31:16];
        16'h018E: mux_lbus_do = blk_dout_reg[15:0];
        16'hFFFC: mux_lbus_do = 16'h4702;
        default:  mux_lbus_do = 16'h0000;
      endcase
   endfunction
endmodule // LBUS_IF
