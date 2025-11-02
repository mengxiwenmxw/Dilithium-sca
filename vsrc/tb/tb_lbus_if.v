`timescale 1ns/1ps

module tb_lbus_if;

    parameter CLK_PERIOD = 10; // 100 MHz

    reg clk_top;
    reg rst_top;

    // input signals
    reg [15:0]      tb_lbus_a ; // Address
    reg [15:0]      tb_lbus_di; // Input data  (Controller -> Cryptographic module)
    reg             tb_lbus_wr; // Assert input data
    reg             tb_lbus_rd; // Assert output data
    wire [15:0]     tb_lbus_do; // Output data (Cryptographic module -> Controller)
    // to hardware 
    wire [23:0]     tb_a;
    wire [23:0]     tb_b;
    wire            tb_blk_krdy; 
    wire            tb_blk_drdy; // blk_drdy assigned to gpio
    wire            tb_blk_en  ; 
    reg             tb_blk_rstn;
    // from hardware
    reg [127:0]     tb_blk_dout;
    reg             tb_blk_kvld;
    reg             tb_blk_dvld;

    LBUS_IF u_LBUS_IF (
        // input signals
        .lbus_a      (tb_lbus_a),
        .lbus_di     (tb_lbus_di),
        .lbus_wr     (tb_lbus_wr),
        .lbus_rd     (tb_lbus_rd),
        .lbus_do     (tb_lbus_do),
        // to hardware 
        .a           (tb_a),
        .b           (tb_b),
        .blk_krdy    (tb_blk_krdy), 
        .blk_drdy    (tb_blk_drdy), 
        .blk_en      (tb_blk_en  ), 
        .blk_rstn    (tb_blk_rstn),
        // from hardware
        .blk_dout    (tb_blk_dout),
        .blk_kvld    (tb_blk_kvld), 
        .blk_dvld    (tb_blk_dvld),

        .clk         (clk_top), 
        .rst         (rst_top)
    );

    // ***** clk & rst *****
    always #((CLK_PERIOD)/2) clk_top = ~clk_top;
    initial begin 
        clk_top = 0;
        rst_top = 1;
        #(CLK_PERIOD * 10);
        rst_top = 0;
    end

    // ***** lbus task *****

    // initialize
    task lbus_init;
        begin
            tb_lbus_a  <= 16'h0000;
            tb_lbus_di <= 16'h0000;
            tb_lbus_wr <= 1'b0;
            tb_lbus_rd <= 1'b1;
        end
    endtask

    // lbus write (takes 3 cycles)
    task lbus_write;
    input [15:0] addr;
    input [15:0] data;
        begin
            @(posedge clk_top);
            tb_lbus_wr <= 1'b1;
            tb_lbus_a <= addr;

            @(posedge clk_top);
            tb_lbus_wr <= 0;
            tb_lbus_di <= data;
            #(CLK_PERIOD * 2);
        end
    endtask

    //lbus read
    task lbus_read;
    input [15:0] addr;
    output [15:0] data;
        begin
            @(posedge clk_top);
            tb_lbus_a <= addr;
            tb_lbus_rd <= 1'b0;

            @(posedge clk_top);
            data <= tb_lbus_do;

            @(posedge clk_top);
            tb_lbus_rd <= 1'b1;
            lbus_init;
        end
    endtask

    // ***** TEST *****
    reg [15:0] read_data;
    initial begin
        $fsdbDumpfile("waves/tb_lbus_if.fsdb");
        $fsdbDumpvars(0, u_LBUS_IF);
        $fsdbDumpMDA(u_LBUS_IF.fifo_a);

        @(negedge rst_top);
        $display("INFO: Reset is released");

        $display("INFO: Initializing...");
        lbus_init;
        #(CLK_PERIOD * 5);

        $display("INFO: Start writing a...");
        lbus_write(16'h0100, 16'haaaa);
        lbus_write(16'h0101, 16'h00bb);
        lbus_write(16'h0102, 16'hcccc);
        lbus_write(16'h0103, 16'h00dd);
        lbus_write(16'h0104, 16'heeee);
        lbus_write(16'h0105, 16'h00ff);
        lbus_write(16'h0106, 16'h1111);
        lbus_write(16'h0107, 16'h0022);
        lbus_write(16'h0108, 16'h3333);
        lbus_write(16'h0109, 16'h0044);
        lbus_write(16'h010a, 16'h5555);
        lbus_write(16'h010b, 16'h0066);

        $display("INFO: Start writing b...");
        lbus_write(16'h0110, 16'h7777);
        lbus_write(16'h0111, 16'h0088);

        $display("INFO: Writing 'start'...");
        lbus_write(16'h0002, 16'h0001);
        
        #(CLK_PERIOD * 20);
        $finish;
    end
endmodule
