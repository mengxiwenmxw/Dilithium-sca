`timescale 1ns/100ps

module tb_chip_sasebo_giii_poly;

reg [15:0]  lbus_di_a;
wire [15:0] lbus_do;
reg         lbus_wrn, lbus_rdn;
reg         lbus_clkn, lbus_rstn;
wire        gpio_startn, gpio_endn, gpio_exec;
wire [9:0]  led;
wire        osc_en_b;

// signals for send
reg [31:0] data_now;
reg wrn_delay;

always @(negedge lbus_clkn or negedge lbus_rstn) begin
    if(~lbus_rstn) begin
        data_now <= 32'd65536;
        wrn_delay <= lbus_wrn;
    end else begin
        wrn_delay <= lbus_wrn;
        if(~wrn_delay&lbus_wrn) data_now <= data_now + 1'b1;
    end
end

// send logic
`define PAUSE_TIME 21'd20
`define DATA_NUM 6'd7

reg [5:0] send_num_cnt;
reg [2:0] send_one_cnt;
reg [20:0] pause_time;


initial lbus_clkn = 0;
always #5 lbus_clkn = ~lbus_clkn;

// FSM
`define SEND_DATA   3'd0
`define SEND_START  3'd1
`define PAUSE       3'd2

`define SEND_H_ADDR 2'b10
`define SEND_H      2'b11
`define SEND_L_ADDR 2'b00
`define SEND_L      2'b01

reg [2:0] state;

// state trans

always @(negedge lbus_clkn or negedge lbus_rstn)begin
    if(~lbus_rstn) begin
        state <= `PAUSE;
        pause_time <= 21'd0;
        send_num_cnt <= 6'd0;
        send_one_cnt <= `SEND_L_ADDR;
    end
    else begin
        case(state)
        `PAUSE:begin
            if(pause_time > `PAUSE_TIME) begin
                state <= `SEND_DATA;
                pause_time <= 21'd0;
                
            end
            else begin 
                state <= `PAUSE; 
                pause_time <= pause_time + 1'b1;
            end
        end
        `SEND_DATA:begin
            if((send_num_cnt >=`DATA_NUM - 1'b1)&&(send_one_cnt == `SEND_H))begin 
                state <= `SEND_START;
                send_num_cnt <= 6'd0;
                send_one_cnt <= `SEND_L_ADDR;
            end else begin 
                state <= `SEND_DATA;
                if(send_one_cnt==`SEND_H)begin
                    send_one_cnt <= `SEND_L_ADDR;
                    send_num_cnt <= send_num_cnt + 1'b1;
                end else begin
                    send_one_cnt <= send_one_cnt + 1'b1;
                end
            end
        end
        `SEND_START:begin
            if(send_one_cnt==`SEND_L)begin
                state <= `PAUSE;
                send_one_cnt <= `SEND_L_ADDR;
            end else begin
                state <= `SEND_START;
                send_one_cnt <= send_one_cnt + 1'b1;
            end
        end
        endcase
    end
end

// data send
always @(negedge lbus_clkn or negedge lbus_rstn)begin
    if(~lbus_rstn)begin
        lbus_di_a <= 16'd0;
        lbus_wrn <= 1'b0;

    end else begin
        case(state)
        `PAUSE:begin
            lbus_di_a <= 16'd0;
            lbus_wrn <= 1'b0;
        end
        `SEND_DATA:begin
            case(send_one_cnt)
            `SEND_L_ADDR:begin
                lbus_wrn <= 1'b1;
                if(send_num_cnt < `DATA_NUM-1'b1)begin
                    lbus_di_a <= 16'h100 + (send_num_cnt<<1);
                end else begin
                    lbus_di_a <= 16'h110;
                end
            end
            `SEND_L:begin
                lbus_wrn <= 1'b0;
                lbus_di_a <= data_now[15:0];
            end
            `SEND_H_ADDR:begin
                lbus_wrn <= 1'b1;
                if(send_num_cnt < `DATA_NUM-1'b1)begin
                    lbus_di_a <= 16'h101 + (send_num_cnt<<1);
                end else begin
                    lbus_di_a <= 16'h111;
                end
            end
            `SEND_H:begin
                lbus_wrn <= 1'b0;
                lbus_di_a <= data_now[31:16];
            end
            endcase
        end
        `SEND_START:begin
            if(send_one_cnt == `SEND_L_ADDR) begin
                lbus_wrn <= 1'b1;
                lbus_di_a <= 16'h002;
            end else begin
                lbus_wrn <= 1'b0;
                lbus_di_a <= 16'h001;
            end
        end
        endcase
    end
end



initial begin
    $fsdbDumpfile("waves/tb_chip_sasebo_giii_poly.fsdb");
    $fsdbDumpvars(0, tb_chip_sasebo_giii_poly);
    lbus_rstn = 0;
    #10 lbus_rstn = 1;
    lbus_rdn = 1;
    #10000;
    $finish;
end

CHIP_SASEBO_GIII_POLY u_top
  (
   .lbus_di_a(lbus_di_a), //地址和数�??????
   .lbus_do(lbus_do),   //
   .lbus_wrn(lbus_wrn), 
   .lbus_rdn(lbus_rdn),
   .lbus_clkn(lbus_clkn), 
   .lbus_rstn(lbus_rstn),


   // GPIO and LED
   .gpio_startn(gpio_startn), 
   .gpio_endn(gpio_endn), 
   .gpio_exec(gpio_exec), 
   .led(led),   //这三个不用管

   // Clock OSC
   .osc_en_b(osc_en_b) //晶振
   );

endmodule