// //SHA3
// `define A_W			10      //1024深度
// `define SHA3512 	2'b00
// `define SHAKE256 	2'b01
// `define SHAKE128 	2'b10
// `define SHA3256 	2'b11
// //SHA3-512
// `define KG1			7'b00_0_0_0_00
// `define KG2			7'b00_1_0_0_01
// //SHA3-256
// `define KH  		7'b11_0_0_0_00
// `define PK			7'b11_1_0_0_01
// `define CT			7'b11_1_0_0_10
// //SHAKE-256
// `define DH			7'b01_0_0_0_00
// `define KDF			7'b01_1_0_0_00
// `define Hw			7'b01_1_0_0_01
// `define CRHt	    7'b01_1_0_0_10
// `define CRHtr	    7'b01_1_0_0_11
// `define CRHK	    7'b01_1_0_1_11
// //Sampler
// `define CBD_0		 7'b01_0_1_0_01
// `define CBD_1		 7'b01_0_1_1_01
// `define ExpandMask	 7'b01_0_1_0_10
// `define ExpandClear	 7'b01_0_1_1_10
// `define SampleInBall 7'b01_0_1_0_11

// Security level
`define DILI_2 2'b01
`define DILI_3 2'b10
`define DILI_5 2'b11

`define KYBER_512 2'b01
`define KYBER_768 2'b10
`define KYBER_1024 2'b11

//NTT mode
`define NTT_256P7L  3'b000
`define NTT_256P8L  3'b001
`define NTT_512P8L  3'b010
`define NTT_512P9L  3'b011
`define INTT_256P7L 3'b100
`define INTT_256P8L 3'b101
`define INTT_512P8L 3'b110
`define INTT_512P9L 3'b111


//POLY
`define KYBER_Q 12'd3329
`define DILI_Q 23'd8380417

`define NTT256  4'b0000
`define NTT512  4'b0001
`define INTT256 4'b0010
`define INTT512 4'b0011
`define PWM     4'b0100
`define MOVE    4'b0101
`define PADD    4'b0110
`define PSUB    4'b0111
`define CSS     4'b1000
`define DECSS   4'b1001
`define DECPS   4'b1010
`define P2R     4'b1011
`define MKHT    4'b1100
`define USHT    4'b1101
`define PWM2    4'b1110 // for kyber

`define css_du_index 24'd2580335
`define css_dv_index 24'd315

`define GAMMA2_2 95232  //(DILI_Q - 1)/88
`define GAMMA2_3 261888 //(DILI_Q - 1)/32
`define GAMMA2_5 261888 //(DILI_Q - 1)/32

`define ALPHA0 18'd190464 //(DILI_Q - 1)/44
`define ALPHA1 19'd523776 //(DILI_Q - 1)/16

`define INDEX0_GAMMA2  24'd11275  // for decompose of GAMMA2 = (Q-1)/88
`define INDEX1_GAMMA2  24'd1025   // for decompose of GAMMA2 = (Q-1)/32

`define USEHINT_m0 24'd44
`define USEHINT_m1 24'd16

`define GAMMA1_2 131072 //2^17
`define BETA_2   78
`define GAMMA1_3 524288 //2^19
`define BETA_3   196
`define GAMMA1_5 524288 //2^19
`define BETA_5   120

`define OMEGA_2 80
`define OMEGA_3 55
`define OMEGA_5 75

`define Z_COMPARE_2 `GAMMA1_2 - `BETA_2
`define Z_COMPARE_3 `GAMMA1_3 - `BETA_3
`define Z_COMPARE_5 `GAMMA1_5 - `BETA_5

`define R_COMPARE_2 `GAMMA2_2 - `BETA_2
`define R_COMPARE_3 `GAMMA2_3 - `BETA_3
`define R_COMPARE_5 `GAMMA2_5 - `BETA_5