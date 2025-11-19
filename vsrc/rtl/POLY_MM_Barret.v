module POLY_MM_Barret(
    input poly_mm_clk,
    input poly_mm_rst_n,
    input poly_mm_enable,
    input [1:0] duv_mode, //00:du=10 01:du=11 10:dv=4 11:dv=5
    input [1:0] compress, //00:no compress 01:compress 11:decompress
    input [1:0] decompose,
    input [23:0] poly_mm_a,
    input [23:0] poly_mm_b,
    input [24:0] poly_mm_m,
    input [4:0]  poly_mm_N,
    input [23:0] poly_mm_q,
    (* dont_touch = "true" *)(* keep = "TRUE" *)output reg poly_mm_valid,
    (* dont_touch = "true" *)(* keep = "TRUE" *)output [23:0] poly_mm_result
    //output reg [23:0] poly_mm_result_s
);

    // reg  [4:0]  poly_mm_N;
    // reg  [4:0]  poly_mm_N_d1, poly_mm_N_d2, poly_mm_N_d3;
    (* dont_touch = "true" *)(* keep = "TRUE" *)reg  [24:0] poly_mm_P0_d_reg0, poly_mm_P0_d_reg1, poly_mm_P0_d_reg2;
    //reg  [23:0] poly_mm_q_d1, poly_mm_q_d2, poly_mm_q_d3;
    //reg  [24:0] poly_mm_m_d1;
    (* dont_touch = "true" *)(* keep = "TRUE" *)wire [47:0] poly_mm_P0;
    (* dont_touch = "true" *)(* keep = "TRUE" *)reg  [24:0] poly_mm_T0;
    (* dont_touch = "true" *)(* keep = "TRUE" *)reg  [24:0] poly_mm_T0_reg;
    (* dont_touch = "true" *)(* keep = "TRUE" *)wire [49:0] poly_mm_P1;
    (* dont_touch = "true" *)(* keep = "TRUE" *)reg  [24:0] poly_mm_T1;
    (* dont_touch = "true" *)(* keep = "TRUE" *)reg  [24:0] poly_mm_T1_reg;
    (* dont_touch = "true" *)(* keep = "TRUE" *)wire [48:0] poly_mm_P2;
    (* dont_touch = "true" *)(* keep = "TRUE" *)reg  [24:0] poly_mm_T2;
    (* dont_touch = "true" *)(* keep = "TRUE" *)reg  [24:0] poly_mm_T2_reg;
    (* dont_touch = "true" *)(* keep = "TRUE" *)wire [24:0] poly_mm_R;
    (* dont_touch = "true" *)(* keep = "TRUE" *)wire [23:0] poly_mm_result_temp;
    (* dont_touch = "true" *)(* keep = "TRUE" *)reg  [2:0]  poly_mm_enable_d;
    (* dont_touch = "true" *)(* keep = "TRUE" *)reg  [24:0] poly_mm_R_mask;


    // ***** Blind Registers *****
    // (* KEEP = "TRUE" *) reg [15:0] lfsr_reg;
    // (* KEEP = "TRUE" *) reg [31:0] dummy_noise_output;
    // (* KEEP = "TRUE" *) reg [23:0] poly_mm_result_dummy;
    // (* KEEP = "TRUE" *) reg [23:0] poly_mm_result_dummy_d0;
    (* dont_touch = "true" *)(* keep = "TRUE" *) reg [47:0] poly_mm_result_reg;// high 24 bit is useful result
    (* dont_touch = "true" *)(* keep = "TRUE" *) wire [23:0] mm_dummy;

    // always @(posedge poly_mm_clk or negedge poly_mm_rst_n) begin
    //     if(!poly_mm_rst_n) begin
    //         poly_mm_N_d1 <= 5'd0;
    //         poly_mm_N_d2 <= 5'd0;
    //         poly_mm_N_d3 <= 5'd0;
    //         poly_mm_q_d1 <= 24'd0;
    //         poly_mm_q_d2 <= 24'd0;
    //         poly_mm_q_d3 <= 24'd0;
    //         poly_mm_m_d1 <= 25'd0;
    //     end else begin
    //         poly_mm_N_d1 <= poly_mm_N;
    //         poly_mm_N_d2 <= poly_mm_N_d1;
    //         poly_mm_N_d3 <= poly_mm_N_d2;
    //         poly_mm_q_d1 <= poly_mm_q;
    //         poly_mm_q_d2 <= poly_mm_q_d1;
    //         poly_mm_q_d3 <= poly_mm_q_d2;
    //         poly_mm_m_d1 <= poly_mm_m;
    //     end
    // end

    // always @(poly_mm_q) begin
    //     casez(poly_mm_q)
    //         24'b1???????????????????????: poly_mm_N = 5'd24;
    //         24'b01??????????????????????: poly_mm_N = 5'd23;
    //         24'b001?????????????????????: poly_mm_N = 5'd22;
    //         24'b0001????????????????????: poly_mm_N = 5'd21;
    //         24'b00001???????????????????: poly_mm_N = 5'd20;
    //         24'b000001??????????????????: poly_mm_N = 5'd19;
    //         24'b0000001?????????????????: poly_mm_N = 5'd18;
    //         24'b00000001????????????????: poly_mm_N = 5'd17;
    //         24'b000000001???????????????: poly_mm_N = 5'd16;
    //         24'b0000000001??????????????: poly_mm_N = 5'd15;
    //         24'b00000000001?????????????: poly_mm_N = 5'd14;
    //         24'b000000000001????????????: poly_mm_N = 5'd13;
    //         24'b0000000000001???????????: poly_mm_N = 5'd12;
    //         24'b00000000000001??????????: poly_mm_N = 5'd11;
    //         24'b000000000000001?????????: poly_mm_N = 5'd10;
    //         24'b0000000000000001????????: poly_mm_N = 5'd9;
    //         24'b00000000000000001???????: poly_mm_N = 5'd8;
    //         24'b000000000000000001??????: poly_mm_N = 5'd7;
    //         24'b0000000000000000001?????: poly_mm_N = 5'd6;
    //         24'b00000000000000000001????: poly_mm_N = 5'd5;
    //         24'b000000000000000000001???: poly_mm_N = 5'd4;
    //         24'b0000000000000000000001??: poly_mm_N = 5'd3;
    //         24'b00000000000000000000001?: poly_mm_N = 5'd2;
    //         24'b000000000000000000000001: poly_mm_N = 5'd1;
    //         default: poly_mm_N = 5'd24;
    //     endcase
    // end

    assign poly_mm_P0 = poly_mm_a * poly_mm_b;
    always @(poly_mm_N or poly_mm_P0) begin
        case(poly_mm_N)
            5'd24: poly_mm_T0 = poly_mm_P0[47:23];
            5'd23: poly_mm_T0 = poly_mm_P0[46:22];
            5'd22: poly_mm_T0 = poly_mm_P0[45:21];
            5'd21: poly_mm_T0 = poly_mm_P0[44:20];
            5'd20: poly_mm_T0 = poly_mm_P0[43:19];
            5'd19: poly_mm_T0 = poly_mm_P0[42:18];
            5'd18: poly_mm_T0 = poly_mm_P0[41:17];
            5'd17: poly_mm_T0 = poly_mm_P0[40:16];
            5'd16: poly_mm_T0 = poly_mm_P0[39:15];
            5'd15: poly_mm_T0 = poly_mm_P0[38:14];
            5'd14: poly_mm_T0 = poly_mm_P0[37:13];
            5'd13: poly_mm_T0 = poly_mm_P0[36:12];
            5'd12: poly_mm_T0 = poly_mm_P0[35:11];
            5'd11: poly_mm_T0 = poly_mm_P0[34:10];
            5'd10: poly_mm_T0 = poly_mm_P0[33:9];
            5'd9:  poly_mm_T0 = poly_mm_P0[32:8];
            5'd8:  poly_mm_T0 = poly_mm_P0[31:7];
            5'd7:  poly_mm_T0 = poly_mm_P0[30:6];
            5'd6:  poly_mm_T0 = poly_mm_P0[29:5];
            5'd5:  poly_mm_T0 = poly_mm_P0[28:4];
            5'd4:  poly_mm_T0 = poly_mm_P0[27:3];
            5'd3:  poly_mm_T0 = poly_mm_P0[26:2];
            5'd2:  poly_mm_T0 = poly_mm_P0[25:1];
            5'd1:  poly_mm_T0 = poly_mm_P0[24:0];
            default: poly_mm_T0 = poly_mm_P0[47:23];
        endcase
    end
    always @(posedge poly_mm_clk or negedge poly_mm_rst_n) begin
        if(!poly_mm_rst_n) begin
            poly_mm_T0_reg <= 25'd0;
        end  else begin
            poly_mm_T0_reg <= poly_mm_T0;
        end
    end

    assign poly_mm_P1 = poly_mm_T0_reg * poly_mm_m;
    always @(poly_mm_N or poly_mm_P1) begin
        case(poly_mm_N)
            5'd24 : poly_mm_T1 = poly_mm_P1[49:25];
            5'd23 : poly_mm_T1 = poly_mm_P1[48:24];
            5'd22 : poly_mm_T1 = poly_mm_P1[47:23];
            5'd21 : poly_mm_T1 = poly_mm_P1[46:22];
            5'd20 : poly_mm_T1 = poly_mm_P1[45:21];
            5'd19 : poly_mm_T1 = poly_mm_P1[44:20];
            5'd18 : poly_mm_T1 = poly_mm_P1[43:19];
            5'd17 : poly_mm_T1 = poly_mm_P1[42:18];
            5'd16 : poly_mm_T1 = poly_mm_P1[41:17];
            5'd15 : poly_mm_T1 = poly_mm_P1[40:16];
            5'd14 : poly_mm_T1 = poly_mm_P1[39:15];
            5'd13 : poly_mm_T1 = poly_mm_P1[38:14];
            5'd12 : poly_mm_T1 = poly_mm_P1[37:13];
            5'd11 : poly_mm_T1 = poly_mm_P1[36:12];
            5'd10 : poly_mm_T1 = poly_mm_P1[35:11];
            5'd9  : poly_mm_T1 = poly_mm_P1[34:10];
            5'd8  : poly_mm_T1 = poly_mm_P1[33:9];
            5'd7  : poly_mm_T1 = poly_mm_P1[32:8];
            5'd6  : poly_mm_T1 = poly_mm_P1[31:7];
            5'd5  : poly_mm_T1 = poly_mm_P1[30:6];
            5'd4  : poly_mm_T1 = poly_mm_P1[29:5];
            5'd3  : poly_mm_T1 = poly_mm_P1[28:4];
            5'd2  : poly_mm_T1 = poly_mm_P1[27:3];
            5'd1  : poly_mm_T1 = poly_mm_P1[26:2];
            default: poly_mm_T1 = poly_mm_P1[49:25];
        endcase
    end

    always @(posedge poly_mm_clk or negedge poly_mm_rst_n) begin
        if(!poly_mm_rst_n) begin
            poly_mm_T1_reg <= 25'd0;
        end else begin
            poly_mm_T1_reg <= poly_mm_T1;
        end
    end

    assign poly_mm_P2 = poly_mm_T1_reg * poly_mm_q;
    always @(poly_mm_N or poly_mm_P2) begin
        case(poly_mm_N)
            5'd24 : poly_mm_T2 = poly_mm_P2[24:0];
            5'd23 : poly_mm_T2 = {1'b0,poly_mm_P2[23:0]};
            5'd22 : poly_mm_T2 = {2'b0,poly_mm_P2[22:0]};
            5'd21 : poly_mm_T2 = {3'b0,poly_mm_P2[21:0]};
            5'd20 : poly_mm_T2 = {4'b0,poly_mm_P2[20:0]};
            5'd19 : poly_mm_T2 = {5'b0,poly_mm_P2[19:0]};
            5'd18 : poly_mm_T2 = {6'b0,poly_mm_P2[18:0]};
            5'd17 : poly_mm_T2 = {7'b0,poly_mm_P2[17:0]};
            5'd16 : poly_mm_T2 = {8'b0,poly_mm_P2[16:0]};
            5'd15 : poly_mm_T2 = {9'b0,poly_mm_P2[15:0]};
            5'd14 : poly_mm_T2 = {10'b0,poly_mm_P2[14:0]};
            5'd13 : poly_mm_T2 = {11'b0,poly_mm_P2[13:0]};
            5'd12 : poly_mm_T2 = {12'b0,poly_mm_P2[12:0]};
            5'd11 : poly_mm_T2 = {13'b0,poly_mm_P2[11:0]};
            5'd10 : poly_mm_T2 = {14'b0,poly_mm_P2[10:0]};
            5'd9  : poly_mm_T2 = {15'b0,poly_mm_P2[9:0]};
            5'd8  : poly_mm_T2 = {16'b0,poly_mm_P2[8:0]};
            5'd7  : poly_mm_T2 = {17'b0,poly_mm_P2[7:0]};
            5'd6  : poly_mm_T2 = {18'b0,poly_mm_P2[6:0]};
            5'd5  : poly_mm_T2 = {19'b0,poly_mm_P2[5:0]};
            5'd4  : poly_mm_T2 = {20'b0,poly_mm_P2[4:0]};
            5'd3  : poly_mm_T2 = {21'b0,poly_mm_P2[3:0]};
            5'd2  : poly_mm_T2 = {22'b0,poly_mm_P2[2:0]};
            5'd1  : poly_mm_T2 = {23'b0,poly_mm_P2[1:0]};
            default: poly_mm_T2 = poly_mm_P2[24:0];
        endcase
    end
    always @(posedge poly_mm_clk or negedge poly_mm_rst_n) begin
        if(!poly_mm_rst_n) begin
            poly_mm_T2_reg <= 25'd0;
        end else begin
            poly_mm_T2_reg <= poly_mm_T2;
        end
    end

    always @(posedge poly_mm_clk or negedge poly_mm_rst_n) begin
        if(!poly_mm_rst_n) begin
            poly_mm_P0_d_reg0 <= 25'd0;
            poly_mm_P0_d_reg1 <= 25'd0;
            poly_mm_P0_d_reg2 <= 25'd0;
        end else begin
            case(poly_mm_N)
                5'd24: poly_mm_P0_d_reg0 <= poly_mm_P0[24:0];
                5'd23: poly_mm_P0_d_reg0 <= {1'b0, poly_mm_P0[23:0]};
                5'd22: poly_mm_P0_d_reg0 <= {2'b0, poly_mm_P0[22:0]};
                5'd21: poly_mm_P0_d_reg0 <= {3'b0, poly_mm_P0[21:0]};
                5'd20: poly_mm_P0_d_reg0 <= {4'b0, poly_mm_P0[20:0]};
                5'd19: poly_mm_P0_d_reg0 <= {5'b0, poly_mm_P0[19:0]};
                5'd18: poly_mm_P0_d_reg0 <= {6'b0, poly_mm_P0[18:0]};
                5'd17: poly_mm_P0_d_reg0 <= {7'b0, poly_mm_P0[17:0]};
                5'd16: poly_mm_P0_d_reg0 <= {8'b0, poly_mm_P0[16:0]};
                5'd15: poly_mm_P0_d_reg0 <= {9'b0, poly_mm_P0[15:0]};
                5'd14: poly_mm_P0_d_reg0 <= {10'b0, poly_mm_P0[14:0]};
                5'd13: poly_mm_P0_d_reg0 <= {11'b0, poly_mm_P0[13:0]};
                5'd12: poly_mm_P0_d_reg0 <= {12'b0, poly_mm_P0[12:0]};
                5'd11: poly_mm_P0_d_reg0 <= {13'b0, poly_mm_P0[11:0]};
                5'd10: poly_mm_P0_d_reg0 <= {14'b0, poly_mm_P0[10:0]};
                5'd9:  poly_mm_P0_d_reg0 <= {15'b0, poly_mm_P0[9:0]};
                5'd8:  poly_mm_P0_d_reg0 <= {16'b0, poly_mm_P0[8:0]};
                5'd7:  poly_mm_P0_d_reg0 <= {17'b0, poly_mm_P0[7:0]};
                5'd6:  poly_mm_P0_d_reg0 <= {18'b0, poly_mm_P0[6:0]};
                5'd5:  poly_mm_P0_d_reg0 <= {19'b0, poly_mm_P0[5:0]};
                5'd4:  poly_mm_P0_d_reg0 <= {20'b0, poly_mm_P0[4:0]};
                5'd3:  poly_mm_P0_d_reg0 <= {21'b0, poly_mm_P0[3:0]};
                5'd2:  poly_mm_P0_d_reg0 <= {22'b0, poly_mm_P0[2:0]};
                5'd1:  poly_mm_P0_d_reg0 <= {23'b0, poly_mm_P0[1:0]};
                default: poly_mm_P0_d_reg0 <= poly_mm_P0[24:0];
            endcase
            poly_mm_P0_d_reg1 <= poly_mm_P0_d_reg0;
            poly_mm_P0_d_reg2 <= poly_mm_P0_d_reg1;
        end
    end

    assign poly_mm_R = poly_mm_P0_d_reg2 - poly_mm_T2_reg;
    always @(poly_mm_N or poly_mm_R_mask) begin
        case(poly_mm_N)
            5'd24: poly_mm_R_mask = 25'h1ffffff;
            5'd23: poly_mm_R_mask = 25'h0ffffff;
            5'd22: poly_mm_R_mask = 25'h07fffff;
            5'd21: poly_mm_R_mask = 25'h03fffff;
            5'd20: poly_mm_R_mask = 25'h01fffff;
            5'd19: poly_mm_R_mask = 25'h00fffff;
            5'd18: poly_mm_R_mask = 25'h007ffff;
            5'd17: poly_mm_R_mask = 25'h003ffff;
            5'd16: poly_mm_R_mask = 25'h001ffff;
            5'd15: poly_mm_R_mask = 25'h000ffff;
            5'd14: poly_mm_R_mask = 25'h0007fff;
            5'd13: poly_mm_R_mask = 25'h0003fff;
            5'd12: poly_mm_R_mask = 25'h0001fff;
            5'd11: poly_mm_R_mask = 25'h0000fff;
            5'd10: poly_mm_R_mask = 25'h00007ff;
            5'd9 : poly_mm_R_mask = 25'h00003ff;
            5'd8 : poly_mm_R_mask = 25'h00001ff;
            5'd7 : poly_mm_R_mask = 25'h00000ff;
            5'd6 : poly_mm_R_mask = 25'h000007f;
            5'd5 : poly_mm_R_mask = 25'h000003f;
            5'd4 : poly_mm_R_mask = 25'h000001f;
            5'd3 : poly_mm_R_mask = 25'h000000f;
            5'd2 : poly_mm_R_mask = 25'h0000007;
            5'd1 : poly_mm_R_mask = 25'h0000003;
            default: poly_mm_R_mask = {1'b0, {24{1'b1}}};
        endcase
    end
    wire [24:0] poly_mm_R_masked = poly_mm_R & poly_mm_R_mask;
    assign poly_mm_result_temp = (poly_mm_R_masked < poly_mm_q) ? poly_mm_R_masked[23:0]: poly_mm_R_masked - poly_mm_q;
    
    // always @(posedge poly_mm_clk or negedge poly_mm_rst_n) begin
    //     if(!poly_mm_rst_n) begin
    //         poly_mm_result <= 24'b0;
    //         // change start 
    //         poly_mm_result_dummy <= 24'b0;
    //         poly_mm_result_dummy_d0 <= 24'b0;
    //         // change end
    //     end
    //     else if(decompose[0] == 1'b1)begin
    //         if(decompose == 2'b01) poly_mm_result <= poly_mm_P0[30] ? poly_mm_P0[45:31] + 1'b1 : poly_mm_P0[45:31] ;
    //         else                   poly_mm_result <= poly_mm_P0[28] ? poly_mm_P0[45:29] + 1'b1 : poly_mm_P0[45:29] ;
    //     end
    //     else if(compress == 2'b01)begin //compress
    //       case(duv_mode) //2'b00:du=10 2'b01:du=11 2'b10:dv=4 2'b11:dv=5
    //         2'b00: //round(12bit * 22'd2580335 >> 23)
    //             poly_mm_result <= {14'b0, (poly_mm_P0[22] ? poly_mm_P0[32:23] + 1'b1 : poly_mm_P0[32:23])};
    //         2'b01: //round(12bit * 22'd2580335 >> 22)
    //             poly_mm_result <= {13'b0, (poly_mm_P0[21] ? poly_mm_P0[32:22] + 1'b1 : poly_mm_P0[32:22])};
    //         2'b10: //round(12bit * 22'd315 >> 16)
    //             poly_mm_result <= {20'b0, (poly_mm_P0[15] ? poly_mm_P0[19:16] + 1'b1 : poly_mm_P0[19:16])};
    //         2'b11: //round(12bit * 22'd315 >> 15)
    //             poly_mm_result <= {19'b0, (poly_mm_P0[14] ? poly_mm_P0[19:15] + 1'b1 : poly_mm_P0[19:15])};
    //       endcase
    //       poly_mm_result_dummy <= mm_dummy;
    //       poly_mm_result_dummy_d0 <= poly_mm_result_dummy;
    //     end
    //     else if(compress == 2'b11)begin //decompress
    //       case(duv_mode) //00:du=10 01:du=11 10:dv=4 11:dv=5
    //         2'b00: //round(12bit × 3329 >> 10)
    //             poly_mm_result <= {12'b0, (poly_mm_P0[9] ? poly_mm_P0[21:10] + 1'b1 : poly_mm_P0[21:10])};
    //         2'b01: //round(12bit × 3329 >> 11)
    //             poly_mm_result <= {12'b0, (poly_mm_P0[10] ? poly_mm_P0[22:11] + 1'b1 : poly_mm_P0[22:11])};
    //         2'b10: //round(12bit × 3329 >> 4)
    //             poly_mm_result <= {12'b0, (poly_mm_P0[3] ? poly_mm_P0[15:4] + 1'b1 : poly_mm_P0[15:4])};
    //         2'b11: //round(12bit × 3329 >> 5)
    //             poly_mm_result <= {12'b0, (poly_mm_P0[4] ? poly_mm_P0[16:5] + 1'b1 : poly_mm_P0[16:5])};
    //       endcase
    //       poly_mm_result_dummy <= mm_dummy;
    //       poly_mm_result_dummy_d0 <= poly_mm_result_dummy;
    //     end
    //     else begin
    //         poly_mm_result <= poly_mm_result_temp;
    //         // change start 
    //         poly_mm_result_dummy <= mm_dummy;
    //         poly_mm_result_dummy_d0 <= poly_mm_result_dummy;
    //         // change end
    //     end
    // end

    // change start
    always @(posedge poly_mm_clk or negedge poly_mm_rst_n) begin
        if(!poly_mm_rst_n) begin
            poly_mm_result_reg <= 48'h000000ffffff;
        end
    //     else if(decompose[0] == 1'b1)begin
    //         if(decompose == 2'b01) poly_mm_result_reg <= poly_mm_P0[30] ? poly_mm_P0[45:31] + 1'b1: poly_mm_P0[45:31] ;
    //         else                   poly_mm_result_reg <= poly_mm_P0[28] ? poly_mm_P0[45:29] + 1'b1 : poly_mm_P0[45:29] ;
    //     end
    //     else if(compress == 2'b01)begin //compress
    //       case(duv_mode) //2'b00:du=10 2'b01:du=11 2'b10:dv=4 2'b11:dv=5
    //         2'b00: //round(12bit * 22'd2580335 >> 23)
    //             poly_mm_result_reg <= {14'b0, (poly_mm_P0[22] ? poly_mm_P0[32:23] + 1'b1 : poly_mm_P0[32:23])};
    //         2'b01: //round(12bit * 22'd2580335 >> 22)
    //             poly_mm_result_reg <= {13'b0, (poly_mm_P0[21] ? poly_mm_P0[32:22] + 1'b1 : poly_mm_P0[32:22])};
    //         2'b10: //round(12bit * 22'd315 >> 16)
    //             poly_mm_result_reg <= {20'b0, (poly_mm_P0[15] ? poly_mm_P0[19:16] + 1'b1 : poly_mm_P0[19:16])};
    //         2'b11: //round(12bit * 22'd315 >> 15)
    //             poly_mm_result_reg <= {19'b0, (poly_mm_P0[14] ? poly_mm_P0[19:15] + 1'b1 : poly_mm_P0[19:15])};
    //       endcase
    //     end
    //     else if(compress == 2'b11)begin //decompress
    //       case(duv_mode) //00:du=10 01:du=11 10:dv=4 11:dv=5
    //         2'b00: //round(12bit × 3329 >> 10)
    //             poly_mm_result_reg <= {12'b0, (poly_mm_P0[9] ? poly_mm_P0[21:10] + 1'b1 : poly_mm_P0[21:10])};
    //         2'b01: //round(12bit × 3329 >> 11)
    //             poly_mm_result_reg <= {12'b0, (poly_mm_P0[10] ? poly_mm_P0[22:11] + 1'b1 : poly_mm_P0[22:11])};
    //         2'b10: //round(12bit × 3329 >> 4)
    //             poly_mm_result_reg <= {12'b0, (poly_mm_P0[3] ? poly_mm_P0[15:4] + 1'b1 : poly_mm_P0[15:4])};
    //         2'b11: //round(12bit × 3329 >> 5)
    //             poly_mm_result_reg <= {12'b0, (poly_mm_P0[4] ? poly_mm_P0[16:5] + 1'b1 : poly_mm_P0[16:5])};
    //       endcase
    //     end
        else begin
            poly_mm_result_reg <= {poly_mm_result_temp,mm_dummy};
        end
    end
    // change end

    // change start 
    assign mm_dummy = (poly_mm_result_reg[47:24]^poly_mm_result_reg[23:0])~^poly_mm_result_temp;
    assign poly_mm_result = poly_mm_result_reg[47:24];
    // change end

    always @(posedge poly_mm_clk or negedge poly_mm_rst_n) begin
        if(!poly_mm_rst_n) begin
            poly_mm_valid <= 1'b0;
            poly_mm_enable_d <= 3'b0;
        end else begin
            {poly_mm_valid, poly_mm_enable_d} <= {poly_mm_enable_d, poly_mm_enable};
        end
    end

    // always @(posedge poly_mm_clk or negedge poly_mm_rst_n) begin 
    //     if(!poly_mm_rst_n) begin
    //         lfsr_reg <= 16'hACE1;   // Random none-zero value
    //     end else if (poly_mm_enable) begin
    //         if (lfsr_reg == 24'h0000 || lfsr_reg[0] === 1'bx) begin
    //             lfsr_reg <= 16'hACE1;
    //         end else begin
    //             lfsr_reg <= {lfsr_reg[14:0], (lfsr_reg[15] ^ lfsr_reg[13] ^ lfsr_reg[10] ^ lfsr_reg[6])};
    //         end
    //     end else begin
    //         lfsr_reg <= lfsr_reg;
    //     end
    // end

    // always @(posedge poly_mm_clk) begin
    //     if(!poly_mm_rst_n) begin
    //         dummy_noise_output <= 32'hAABBCCDD;   // Random none-zero value
    //     end else if(poly_mm_enable) begin
    //         dummy_noise_output <= {dummy_noise_output[15:0], (lfsr_reg[7:0] * lfsr_reg[15:8])};
    //     end else begin
    //         dummy_noise_output <= dummy_noise_output;
    //     end
    // end

endmodule