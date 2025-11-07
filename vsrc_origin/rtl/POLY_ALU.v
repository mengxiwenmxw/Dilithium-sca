module POLY_ALU(
    input poly_clk,
    input poly_rst_n,
    input poly_enable,
    input [9:0] poly_mode,
    input [1:0] poly_decompose,
    input [1:0] poly_compress,
    input [1:0] poly_duv_mode,
    input [23:0] poly_data_in0,
    input [23:0] poly_data_in1,
    input [23:0] poly_data_in2,
    input [23:0] poly_data_in3,
    input [23:0] poly_q,
    input [24:0] poly_barret_m,
    input [4:0] poly_mm_N,
    output       poly_valid,
    (* dont_touch = "true" *)(* keep = "TRUE" *) output reg [23:0] poly_data_out0,
    (* dont_touch = "true" *)(* keep = "TRUE" *) output reg [23:0] poly_data_out1
);

    wire [23:0] poly_div2_a_dout, poly_div2_b_dout;
    wire [23:0] poly_ma_dout, poly_ms_dout, poly_mm_dout;
    wire [23:0] poly_ms_a, poly_ms_b;
    wire [23:0] poly_ma_a, poly_ma_b;
    (* dont_touch = "true" *)(* keep = "TRUE" *) reg  [23:0] poly_mm_dina, poly_mm_dinb;
    reg         poly_mm_enable_reg;
    wire        poly_mm_valid;
    reg  [4:0]  poly_valid_reg;

    always @(posedge poly_clk or negedge poly_rst_n) begin
        if (!poly_rst_n) begin
            poly_mm_dina <= 24'd0;
            poly_mm_dinb <= 24'd0;
            poly_mm_enable_reg <= 1'b0;
        end else if (poly_enable) begin
            poly_mm_dina <= poly_data_in0;
            poly_mm_dinb <= poly_mode[8] ? poly_data_in1 : poly_ms_dout;
            poly_mm_enable_reg <= poly_enable;
        end
    end


    reg [23:0] poly_delay_reg_0, poly_delay_reg_1, poly_delay_reg_2, poly_delay_reg_3, poly_delay_reg_4;
    always @(posedge poly_clk or negedge poly_rst_n) begin
        if (!poly_rst_n) begin
            poly_delay_reg_0 <= 24'd0;
            poly_delay_reg_1 <= 24'd0;
            poly_delay_reg_2 <= 24'd0;
            poly_delay_reg_3 <= 24'd0;
            poly_delay_reg_4 <= 24'd0;
            poly_valid_reg <= 5'b0;
        end begin
            poly_delay_reg_0 <= poly_mode[9] ? poly_ma_dout : poly_data_in2;
            poly_delay_reg_1 <= poly_delay_reg_0;
            poly_delay_reg_2 <= poly_delay_reg_1;
            poly_delay_reg_3 <= poly_delay_reg_2;
            poly_delay_reg_4 <= poly_delay_reg_3;
            poly_valid_reg <= {poly_valid_reg[3:0], poly_enable};
        end
    end

    (* dont_touch = "true" *)(* keep = "TRUE" *) POLY_MM_Barret u_POLY_MM_Barret(
        .poly_mm_clk    ( poly_clk    ),
        .poly_mm_rst_n  ( poly_rst_n  ),
        .poly_mm_enable ( poly_mm_enable_reg ),
        .decompose      ( poly_decompose ),
        .duv_mode       ( poly_duv_mode  ),
        .compress       ( poly_compress  ),
        .poly_mm_a      ( poly_mm_dina      ),
        .poly_mm_b      ( poly_mm_dinb      ),
        .poly_mm_m      ( poly_barret_m      ),
        .poly_mm_q      ( poly_q      ),
        .poly_mm_N      ( poly_mm_N      ),
        .poly_mm_valid  ( poly_mm_valid  ),
        .poly_mm_result  ( poly_mm_dout  )
    );


    assign poly_ms_a = poly_mode[6] ? (poly_mode[7] ? poly_delay_reg_4 : poly_data_in2)
                                    : poly_delay_reg_4;
    assign poly_ms_b = poly_mode[4] ? poly_data_in3 : poly_mm_dout;
    (* dont_touch = "true" *)(* keep = "TRUE" *) POLY_MS u_POLY_MS(
        .poly_ms_a ( poly_ms_a ),
        .poly_ms_b ( poly_ms_b ),
        .poly_ms_q ( poly_q    ),
        .poly_q_width( poly_mm_N),
        .poly_ms_o ( poly_ms_dout  )
    );

    assign poly_ma_a = poly_mode[7] ? poly_delay_reg_4 : poly_data_in2;
    assign poly_ma_b = poly_mode[5] ? poly_data_in3 : poly_mm_dout;
    (* dont_touch = "true" *)(* keep = "TRUE" *) POLY_MA u_POLY_MA(
        .poly_ma_a ( poly_ma_a ),
        .poly_ma_b ( poly_ma_b ),
        .poly_ma_q ( poly_q ),
        .poly_ma_o ( poly_ma_dout  )
    );

    (* dont_touch = "true" *)(* keep = "TRUE" *) POLY_DIV2 u_POLY_DIV2_a(
        .poly_div2_din ( poly_delay_reg_4 ),
        .poly_div2_q   ( poly_q   ),
        .poly_div2_dout  ( poly_div2_a_dout  )
    );

    (* dont_touch = "true" *)(* keep = "TRUE" *) POLY_DIV2 u_POLY_DIV2_b(
        .poly_div2_din ( poly_mm_dout ),
        .poly_div2_q   ( poly_q   ),
        .poly_div2_dout  ( poly_div2_b_dout  )
    );


    always @(*) begin
        case(poly_mode[3:2])
            2'b00: poly_data_out0 = poly_div2_a_dout;
            2'b01: poly_data_out0 = poly_delay_reg_4;
            2'b11: poly_data_out0 = poly_ma_dout;
            default: poly_data_out0 = poly_div2_a_dout;
        endcase
    end

    always @(*) begin
        case(poly_mode[1:0])
            2'b00: poly_data_out1 = poly_ms_dout;
            2'b01: poly_data_out1 = poly_mm_dout;
            2'b11: poly_data_out1 = poly_div2_b_dout;
            default: poly_data_out1 = poly_ms_dout;
        endcase
    end

    assign poly_valid = poly_valid_reg[4];

endmodule