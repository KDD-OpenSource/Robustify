
Ö
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
7
Square
x"T
y"T"
Ttype:
2	
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68 
z
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

:*
dtype0
~
net_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namenet_output/kernel
w
%net_output/kernel/Read/ReadVariableOpReadVariableOpnet_output/kernel*
_output_shapes

:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_11/kernel/m

*Adam/dense_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/m*
_output_shapes

:*
dtype0

Adam/net_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/net_output/kernel/m

,Adam/net_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/net_output/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_11/kernel/v

*Adam/dense_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/v*
_output_shapes

:*
dtype0

Adam/net_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/net_output/kernel/v

,Adam/net_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/net_output/kernel/v*
_output_shapes

:*
dtype0

NoOpNoOp
¦ 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*á
value×BÔ BÍ
ÿ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

	optimizer
loss
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 


kernel
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*


kernel
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses*

"	keras_api* 

#	keras_api* 

$	keras_api* 

%	keras_api* 

&	keras_api* 

'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses* 
h
-iter

.beta_1

/beta_2
	0decay
1learning_ratemPmQvRvS*
* 

0
1*

0
1*
* 
°
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

7serving_default* 
_Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 
°
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
__call__
=activity_regularizer_fn
*&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*
* 
* 
a[
VARIABLE_VALUEnet_output/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 
°
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
Dactivity_regularizer_fn
*!&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
C
0
1
2
3
4
5
6
7
	8*

K0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	Ltotal
	Mcount
N	variables
O	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

L0
M1*

N	variables*
|
VARIABLE_VALUEAdam/dense_11/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/net_output/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_11/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/net_output/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
serving_default_input_12Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
Ö
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_12dense_11/kernelnet_output/kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_62338
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_11/kernel/Read/ReadVariableOp%net_output/kernel/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_11/kernel/m/Read/ReadVariableOp,Adam/net_output/kernel/m/Read/ReadVariableOp*Adam/dense_11/kernel/v/Read/ReadVariableOp,Adam/net_output/kernel/v/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_62459
ý
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_11/kernelnet_output/kernel	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_11/kernel/mAdam/net_output/kernel/mAdam/dense_11/kernel/vAdam/net_output/kernel/v*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_62508ÁÙ


#__inference_signature_wrapper_62338
input_12
unknown:
	unknown_0:
identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinput_12unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_61892k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_12

®
E__inference_net_output_layer_call_and_return_conditional_losses_61953

inputs0
matmul_readvariableop_resource:
identity¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·
r
F__inference_add_loss_11_layer_call_and_return_conditional_losses_62381

inputs
identity

identity_1=
IdentityIdentityinputs*
T0*
_output_shapes
: ?

Identity_1Identityinputs*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
Á7
Î
C__inference_model_29_layer_call_and_return_conditional_losses_62327

inputs9
'dense_11_matmul_readvariableop_resource:;
)net_output_matmul_readvariableop_resource:
identity

identity_1

identity_2

identity_3¢dense_11/MatMul/ReadVariableOp¢ net_output/MatMul/ReadVariableOp
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype0{
dense_11/MatMulMatMulinputs&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense_11/ReluReludense_11/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
#dense_11/ActivityRegularizer/SquareSquaredense_11/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
"dense_11/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_11/ActivityRegularizer/SumSum'dense_11/ActivityRegularizer/Square:y:0+dense_11/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_11/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *     
 dense_11/ActivityRegularizer/mulMul+dense_11/ActivityRegularizer/mul/x:output:0)dense_11/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: m
"dense_11/ActivityRegularizer/ShapeShapedense_11/Relu:activations:0*
T0*
_output_shapes
:z
0dense_11/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dense_11/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dense_11/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
*dense_11/ActivityRegularizer/strided_sliceStridedSlice+dense_11/ActivityRegularizer/Shape:output:09dense_11/ActivityRegularizer/strided_slice/stack:output:0;dense_11/ActivityRegularizer/strided_slice/stack_1:output:0;dense_11/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
!dense_11/ActivityRegularizer/CastCast3dense_11/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
$dense_11/ActivityRegularizer/truedivRealDiv$dense_11/ActivityRegularizer/mul:z:0%dense_11/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
 net_output/MatMul/ReadVariableOpReadVariableOp)net_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
net_output/MatMulMatMuldense_11/Relu:activations:0(net_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
net_output/ReluRelunet_output/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%net_output/ActivityRegularizer/SquareSquarenet_output/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
$net_output/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¤
"net_output/ActivityRegularizer/SumSum)net_output/ActivityRegularizer/Square:y:0-net_output/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: i
$net_output/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ¦
"net_output/ActivityRegularizer/mulMul-net_output/ActivityRegularizer/mul/x:output:0+net_output/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: q
$net_output/ActivityRegularizer/ShapeShapenet_output/Relu:activations:0*
T0*
_output_shapes
:|
2net_output/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4net_output/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4net_output/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
,net_output/ActivityRegularizer/strided_sliceStridedSlice-net_output/ActivityRegularizer/Shape:output:0;net_output/ActivityRegularizer/strided_slice/stack:output:0=net_output/ActivityRegularizer/strided_slice/stack_1:output:0=net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
#net_output/ActivityRegularizer/CastCast5net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: £
&net_output/ActivityRegularizer/truedivRealDiv&net_output/ActivityRegularizer/mul:z:0'net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
tf.math.subtract_11/Sub/yConst*
_output_shapes
:*
dtype0*Q
valueHBF"<ÍÌÌ=    ÍÌÌ=ÍÌÌ=ÍÌÌ=    ÍÌÌ=ÍÌÌ=ÍÌÌ="þ¾>ÍÌÌ=ÍÌÌ=&¬~>    ÍÌÌ=
tf.math.subtract_11/SubSubnet_output/Relu:activations:0"tf.math.subtract_11/Sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
tf.math.pow_11/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
tf.math.pow_11/PowPowtf.math.subtract_11/Sub:z:0tf.math.pow_11/Pow/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
+tf.math.reduce_sum_11/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
tf.math.reduce_sum_11/SumSumtf.math.pow_11/Pow:z:04tf.math.reduce_sum_11/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
tf.math.reduce_mean_11/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
tf.math.reduce_mean_11/MeanMean"tf.math.reduce_sum_11/Sum:output:0%tf.math.reduce_mean_11/Const:output:0*
T0*
_output_shapes
: ^
tf.__operators__.add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *7
tf.__operators__.add_11/AddV2AddV2$tf.math.reduce_mean_11/Mean:output:0"tf.__operators__.add_11/y:output:0*
T0*
_output_shapes
: m
IdentityIdentity"tf.math.reduce_sum_11/Sum:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿh

Identity_1Identity(dense_11/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: j

Identity_2Identity*net_output/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: a

Identity_3Identity!tf.__operators__.add_11/AddV2:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp^dense_11/MatMul/ReadVariableOp!^net_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2D
 net_output/MatMul/ReadVariableOp net_output/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Å

(__inference_model_29_layer_call_fn_62229

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: : : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_29_layer_call_and_return_conditional_losses_62095k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

®
E__inference_net_output_layer_call_and_return_conditional_losses_62397

inputs0
matmul_readvariableop_resource:
identity¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë

(__inference_model_29_layer_call_fn_62117
input_12
unknown:
	unknown_0:
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinput_12unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: : : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_29_layer_call_and_return_conditional_losses_62095k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_12

¬
C__inference_dense_11_layer_call_and_return_conditional_losses_62389

inputs0
matmul_readvariableop_resource:
identity¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë

(__inference_model_29_layer_call_fn_61997
input_12
unknown:
	unknown_0:
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinput_12unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: : : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_29_layer_call_and_return_conditional_losses_61987k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_12
Á7
Î
C__inference_model_29_layer_call_and_return_conditional_losses_62278

inputs9
'dense_11_matmul_readvariableop_resource:;
)net_output_matmul_readvariableop_resource:
identity

identity_1

identity_2

identity_3¢dense_11/MatMul/ReadVariableOp¢ net_output/MatMul/ReadVariableOp
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype0{
dense_11/MatMulMatMulinputs&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense_11/ReluReludense_11/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
#dense_11/ActivityRegularizer/SquareSquaredense_11/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
"dense_11/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_11/ActivityRegularizer/SumSum'dense_11/ActivityRegularizer/Square:y:0+dense_11/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_11/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *     
 dense_11/ActivityRegularizer/mulMul+dense_11/ActivityRegularizer/mul/x:output:0)dense_11/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: m
"dense_11/ActivityRegularizer/ShapeShapedense_11/Relu:activations:0*
T0*
_output_shapes
:z
0dense_11/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dense_11/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dense_11/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
*dense_11/ActivityRegularizer/strided_sliceStridedSlice+dense_11/ActivityRegularizer/Shape:output:09dense_11/ActivityRegularizer/strided_slice/stack:output:0;dense_11/ActivityRegularizer/strided_slice/stack_1:output:0;dense_11/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
!dense_11/ActivityRegularizer/CastCast3dense_11/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
$dense_11/ActivityRegularizer/truedivRealDiv$dense_11/ActivityRegularizer/mul:z:0%dense_11/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
 net_output/MatMul/ReadVariableOpReadVariableOp)net_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
net_output/MatMulMatMuldense_11/Relu:activations:0(net_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
net_output/ReluRelunet_output/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%net_output/ActivityRegularizer/SquareSquarenet_output/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
$net_output/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¤
"net_output/ActivityRegularizer/SumSum)net_output/ActivityRegularizer/Square:y:0-net_output/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: i
$net_output/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ¦
"net_output/ActivityRegularizer/mulMul-net_output/ActivityRegularizer/mul/x:output:0+net_output/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: q
$net_output/ActivityRegularizer/ShapeShapenet_output/Relu:activations:0*
T0*
_output_shapes
:|
2net_output/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4net_output/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4net_output/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
,net_output/ActivityRegularizer/strided_sliceStridedSlice-net_output/ActivityRegularizer/Shape:output:0;net_output/ActivityRegularizer/strided_slice/stack:output:0=net_output/ActivityRegularizer/strided_slice/stack_1:output:0=net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
#net_output/ActivityRegularizer/CastCast5net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: £
&net_output/ActivityRegularizer/truedivRealDiv&net_output/ActivityRegularizer/mul:z:0'net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
tf.math.subtract_11/Sub/yConst*
_output_shapes
:*
dtype0*Q
valueHBF"<ÍÌÌ=    ÍÌÌ=ÍÌÌ=ÍÌÌ=    ÍÌÌ=ÍÌÌ=ÍÌÌ="þ¾>ÍÌÌ=ÍÌÌ=&¬~>    ÍÌÌ=
tf.math.subtract_11/SubSubnet_output/Relu:activations:0"tf.math.subtract_11/Sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
tf.math.pow_11/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
tf.math.pow_11/PowPowtf.math.subtract_11/Sub:z:0tf.math.pow_11/Pow/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
+tf.math.reduce_sum_11/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
tf.math.reduce_sum_11/SumSumtf.math.pow_11/Pow:z:04tf.math.reduce_sum_11/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
tf.math.reduce_mean_11/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
tf.math.reduce_mean_11/MeanMean"tf.math.reduce_sum_11/Sum:output:0%tf.math.reduce_mean_11/Const:output:0*
T0*
_output_shapes
: ^
tf.__operators__.add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *7
tf.__operators__.add_11/AddV2AddV2$tf.math.reduce_mean_11/Mean:output:0"tf.__operators__.add_11/y:output:0*
T0*
_output_shapes
: m
IdentityIdentity"tf.math.reduce_sum_11/Sum:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿh

Identity_1Identity(dense_11/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: j

Identity_2Identity*net_output/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: a

Identity_3Identity!tf.__operators__.add_11/AddV2:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp^dense_11/MatMul/ReadVariableOp!^net_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2D
 net_output/MatMul/ReadVariableOp net_output/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò$
Ù
__inference__traced_save_62459
file_prefix.
*savev2_dense_11_kernel_read_readvariableop0
,savev2_net_output_kernel_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_11_kernel_m_read_readvariableop7
3savev2_adam_net_output_kernel_m_read_readvariableop5
1savev2_adam_dense_11_kernel_v_read_readvariableop7
3savev2_adam_net_output_kernel_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ä
valueºB·B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B ÷
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_11_kernel_read_readvariableop,savev2_net_output_kernel_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_11_kernel_m_read_readvariableop3savev2_adam_net_output_kernel_m_read_readvariableop1savev2_adam_dense_11_kernel_v_read_readvariableop3savev2_adam_net_output_kernel_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*a
_input_shapesP
N: ::: : : : : : : ::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :$
 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: 
ð

­
I__inference_net_output_layer_call_and_return_all_conditional_losses_62370

inputs
unknown:
identity

identity_1¢StatefulPartitionedCallÍ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_net_output_layer_call_and_return_conditional_losses_61953¥
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *:
f5R3
1__inference_net_output_activity_regularizer_61918o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù4
¢
C__inference_model_29_layer_call_and_return_conditional_losses_62199
input_12 
dense_11_62161:"
net_output_62172:
identity

identity_1

identity_2

identity_3¢ dense_11/StatefulPartitionedCall¢"net_output/StatefulPartitionedCallÝ
 dense_11/StatefulPartitionedCallStatefulPartitionedCallinput_12dense_11_62161*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_61933É
,dense_11/ActivityRegularizer/PartitionedCallPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *8
f3R1
/__inference_dense_11_activity_regularizer_61905{
"dense_11/ActivityRegularizer/ShapeShape)dense_11/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0dense_11/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dense_11/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dense_11/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
*dense_11/ActivityRegularizer/strided_sliceStridedSlice+dense_11/ActivityRegularizer/Shape:output:09dense_11/ActivityRegularizer/strided_slice/stack:output:0;dense_11/ActivityRegularizer/strided_slice/stack_1:output:0;dense_11/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
!dense_11/ActivityRegularizer/CastCast3dense_11/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ®
$dense_11/ActivityRegularizer/truedivRealDiv5dense_11/ActivityRegularizer/PartitionedCall:output:0%dense_11/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
"net_output/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0net_output_62172*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_net_output_layer_call_and_return_conditional_losses_61953Ï
.net_output/ActivityRegularizer/PartitionedCallPartitionedCall+net_output/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *:
f5R3
1__inference_net_output_activity_regularizer_61918
$net_output/ActivityRegularizer/ShapeShape+net_output/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:|
2net_output/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4net_output/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4net_output/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
,net_output/ActivityRegularizer/strided_sliceStridedSlice-net_output/ActivityRegularizer/Shape:output:0;net_output/ActivityRegularizer/strided_slice/stack:output:0=net_output/ActivityRegularizer/strided_slice/stack_1:output:0=net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
#net_output/ActivityRegularizer/CastCast5net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ´
&net_output/ActivityRegularizer/truedivRealDiv7net_output/ActivityRegularizer/PartitionedCall:output:0'net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
tf.math.subtract_11/Sub/yConst*
_output_shapes
:*
dtype0*Q
valueHBF"<ÍÌÌ=    ÍÌÌ=ÍÌÌ=ÍÌÌ=    ÍÌÌ=ÍÌÌ=ÍÌÌ="þ¾>ÍÌÌ=ÍÌÌ=&¬~>    ÍÌÌ=¡
tf.math.subtract_11/SubSub+net_output/StatefulPartitionedCall:output:0"tf.math.subtract_11/Sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
tf.math.pow_11/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
tf.math.pow_11/PowPowtf.math.subtract_11/Sub:z:0tf.math.pow_11/Pow/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
+tf.math.reduce_sum_11/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
tf.math.reduce_sum_11/SumSumtf.math.pow_11/Pow:z:04tf.math.reduce_sum_11/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
tf.math.reduce_mean_11/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
tf.math.reduce_mean_11/MeanMean"tf.math.reduce_sum_11/Sum:output:0%tf.math.reduce_mean_11/Const:output:0*
T0*
_output_shapes
: ^
tf.__operators__.add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *7
tf.__operators__.add_11/AddV2AddV2$tf.math.reduce_mean_11/Mean:output:0"tf.__operators__.add_11/y:output:0*
T0*
_output_shapes
: Ê
add_loss_11/PartitionedCallPartitionedCall!tf.__operators__.add_11/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_add_loss_11_layer_call_and_return_conditional_losses_61980m
IdentityIdentity"tf.math.reduce_sum_11/Sum:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿh

Identity_1Identity(dense_11/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: j

Identity_2Identity*net_output/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: d

Identity_3Identity$add_loss_11/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: 
NoOpNoOp!^dense_11/StatefulPartitionedCall#^net_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2H
"net_output/StatefulPartitionedCall"net_output/StatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_12

¬
C__inference_dense_11_layer_call_and_return_conditional_losses_61933

inputs0
matmul_readvariableop_resource:
identity¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó4
 
C__inference_model_29_layer_call_and_return_conditional_losses_62095

inputs 
dense_11_62057:"
net_output_62068:
identity

identity_1

identity_2

identity_3¢ dense_11/StatefulPartitionedCall¢"net_output/StatefulPartitionedCallÛ
 dense_11/StatefulPartitionedCallStatefulPartitionedCallinputsdense_11_62057*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_61933É
,dense_11/ActivityRegularizer/PartitionedCallPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *8
f3R1
/__inference_dense_11_activity_regularizer_61905{
"dense_11/ActivityRegularizer/ShapeShape)dense_11/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0dense_11/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dense_11/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dense_11/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
*dense_11/ActivityRegularizer/strided_sliceStridedSlice+dense_11/ActivityRegularizer/Shape:output:09dense_11/ActivityRegularizer/strided_slice/stack:output:0;dense_11/ActivityRegularizer/strided_slice/stack_1:output:0;dense_11/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
!dense_11/ActivityRegularizer/CastCast3dense_11/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ®
$dense_11/ActivityRegularizer/truedivRealDiv5dense_11/ActivityRegularizer/PartitionedCall:output:0%dense_11/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
"net_output/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0net_output_62068*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_net_output_layer_call_and_return_conditional_losses_61953Ï
.net_output/ActivityRegularizer/PartitionedCallPartitionedCall+net_output/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *:
f5R3
1__inference_net_output_activity_regularizer_61918
$net_output/ActivityRegularizer/ShapeShape+net_output/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:|
2net_output/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4net_output/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4net_output/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
,net_output/ActivityRegularizer/strided_sliceStridedSlice-net_output/ActivityRegularizer/Shape:output:0;net_output/ActivityRegularizer/strided_slice/stack:output:0=net_output/ActivityRegularizer/strided_slice/stack_1:output:0=net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
#net_output/ActivityRegularizer/CastCast5net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ´
&net_output/ActivityRegularizer/truedivRealDiv7net_output/ActivityRegularizer/PartitionedCall:output:0'net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
tf.math.subtract_11/Sub/yConst*
_output_shapes
:*
dtype0*Q
valueHBF"<ÍÌÌ=    ÍÌÌ=ÍÌÌ=ÍÌÌ=    ÍÌÌ=ÍÌÌ=ÍÌÌ="þ¾>ÍÌÌ=ÍÌÌ=&¬~>    ÍÌÌ=¡
tf.math.subtract_11/SubSub+net_output/StatefulPartitionedCall:output:0"tf.math.subtract_11/Sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
tf.math.pow_11/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
tf.math.pow_11/PowPowtf.math.subtract_11/Sub:z:0tf.math.pow_11/Pow/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
+tf.math.reduce_sum_11/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
tf.math.reduce_sum_11/SumSumtf.math.pow_11/Pow:z:04tf.math.reduce_sum_11/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
tf.math.reduce_mean_11/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
tf.math.reduce_mean_11/MeanMean"tf.math.reduce_sum_11/Sum:output:0%tf.math.reduce_mean_11/Const:output:0*
T0*
_output_shapes
: ^
tf.__operators__.add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *7
tf.__operators__.add_11/AddV2AddV2$tf.math.reduce_mean_11/Mean:output:0"tf.__operators__.add_11/y:output:0*
T0*
_output_shapes
: Ê
add_loss_11/PartitionedCallPartitionedCall!tf.__operators__.add_11/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_add_loss_11_layer_call_and_return_conditional_losses_61980m
IdentityIdentity"tf.math.reduce_sum_11/Sum:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿh

Identity_1Identity(dense_11/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: j

Identity_2Identity*net_output/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: d

Identity_3Identity$add_loss_11/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: 
NoOpNoOp!^dense_11/StatefulPartitionedCall#^net_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2H
"net_output/StatefulPartitionedCall"net_output/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Å

(__inference_model_29_layer_call_fn_62217

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: : : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_29_layer_call_and_return_conditional_losses_61987k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó4
 
C__inference_model_29_layer_call_and_return_conditional_losses_61987

inputs 
dense_11_61934:"
net_output_61954:
identity

identity_1

identity_2

identity_3¢ dense_11/StatefulPartitionedCall¢"net_output/StatefulPartitionedCallÛ
 dense_11/StatefulPartitionedCallStatefulPartitionedCallinputsdense_11_61934*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_61933É
,dense_11/ActivityRegularizer/PartitionedCallPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *8
f3R1
/__inference_dense_11_activity_regularizer_61905{
"dense_11/ActivityRegularizer/ShapeShape)dense_11/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0dense_11/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dense_11/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dense_11/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
*dense_11/ActivityRegularizer/strided_sliceStridedSlice+dense_11/ActivityRegularizer/Shape:output:09dense_11/ActivityRegularizer/strided_slice/stack:output:0;dense_11/ActivityRegularizer/strided_slice/stack_1:output:0;dense_11/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
!dense_11/ActivityRegularizer/CastCast3dense_11/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ®
$dense_11/ActivityRegularizer/truedivRealDiv5dense_11/ActivityRegularizer/PartitionedCall:output:0%dense_11/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
"net_output/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0net_output_61954*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_net_output_layer_call_and_return_conditional_losses_61953Ï
.net_output/ActivityRegularizer/PartitionedCallPartitionedCall+net_output/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *:
f5R3
1__inference_net_output_activity_regularizer_61918
$net_output/ActivityRegularizer/ShapeShape+net_output/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:|
2net_output/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4net_output/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4net_output/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
,net_output/ActivityRegularizer/strided_sliceStridedSlice-net_output/ActivityRegularizer/Shape:output:0;net_output/ActivityRegularizer/strided_slice/stack:output:0=net_output/ActivityRegularizer/strided_slice/stack_1:output:0=net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
#net_output/ActivityRegularizer/CastCast5net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ´
&net_output/ActivityRegularizer/truedivRealDiv7net_output/ActivityRegularizer/PartitionedCall:output:0'net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
tf.math.subtract_11/Sub/yConst*
_output_shapes
:*
dtype0*Q
valueHBF"<ÍÌÌ=    ÍÌÌ=ÍÌÌ=ÍÌÌ=    ÍÌÌ=ÍÌÌ=ÍÌÌ="þ¾>ÍÌÌ=ÍÌÌ=&¬~>    ÍÌÌ=¡
tf.math.subtract_11/SubSub+net_output/StatefulPartitionedCall:output:0"tf.math.subtract_11/Sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
tf.math.pow_11/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
tf.math.pow_11/PowPowtf.math.subtract_11/Sub:z:0tf.math.pow_11/Pow/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
+tf.math.reduce_sum_11/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
tf.math.reduce_sum_11/SumSumtf.math.pow_11/Pow:z:04tf.math.reduce_sum_11/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
tf.math.reduce_mean_11/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
tf.math.reduce_mean_11/MeanMean"tf.math.reduce_sum_11/Sum:output:0%tf.math.reduce_mean_11/Const:output:0*
T0*
_output_shapes
: ^
tf.__operators__.add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *7
tf.__operators__.add_11/AddV2AddV2$tf.math.reduce_mean_11/Mean:output:0"tf.__operators__.add_11/y:output:0*
T0*
_output_shapes
: Ê
add_loss_11/PartitionedCallPartitionedCall!tf.__operators__.add_11/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_add_loss_11_layer_call_and_return_conditional_losses_61980m
IdentityIdentity"tf.math.reduce_sum_11/Sum:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿh

Identity_1Identity(dense_11/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: j

Identity_2Identity*net_output/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: d

Identity_3Identity$add_loss_11/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: 
NoOpNoOp!^dense_11/StatefulPartitionedCall#^net_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2H
"net_output/StatefulPartitionedCall"net_output/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

~
*__inference_net_output_layer_call_fn_62361

inputs
unknown:
identity¢StatefulPartitionedCallÍ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_net_output_layer_call_and_return_conditional_losses_61953o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

|
(__inference_dense_11_layer_call_fn_62345

inputs
unknown:
identity¢StatefulPartitionedCallË
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_61933o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

H
1__inference_net_output_activity_regularizer_61918
x
identity6
SquareSquarex*
T0*
_output_shapes
:9
RankRank
Square:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
ù4
¢
C__inference_model_29_layer_call_and_return_conditional_losses_62158
input_12 
dense_11_62120:"
net_output_62131:
identity

identity_1

identity_2

identity_3¢ dense_11/StatefulPartitionedCall¢"net_output/StatefulPartitionedCallÝ
 dense_11/StatefulPartitionedCallStatefulPartitionedCallinput_12dense_11_62120*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_61933É
,dense_11/ActivityRegularizer/PartitionedCallPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *8
f3R1
/__inference_dense_11_activity_regularizer_61905{
"dense_11/ActivityRegularizer/ShapeShape)dense_11/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0dense_11/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dense_11/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dense_11/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
*dense_11/ActivityRegularizer/strided_sliceStridedSlice+dense_11/ActivityRegularizer/Shape:output:09dense_11/ActivityRegularizer/strided_slice/stack:output:0;dense_11/ActivityRegularizer/strided_slice/stack_1:output:0;dense_11/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
!dense_11/ActivityRegularizer/CastCast3dense_11/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ®
$dense_11/ActivityRegularizer/truedivRealDiv5dense_11/ActivityRegularizer/PartitionedCall:output:0%dense_11/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
"net_output/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0net_output_62131*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_net_output_layer_call_and_return_conditional_losses_61953Ï
.net_output/ActivityRegularizer/PartitionedCallPartitionedCall+net_output/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *:
f5R3
1__inference_net_output_activity_regularizer_61918
$net_output/ActivityRegularizer/ShapeShape+net_output/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:|
2net_output/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4net_output/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4net_output/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
,net_output/ActivityRegularizer/strided_sliceStridedSlice-net_output/ActivityRegularizer/Shape:output:0;net_output/ActivityRegularizer/strided_slice/stack:output:0=net_output/ActivityRegularizer/strided_slice/stack_1:output:0=net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
#net_output/ActivityRegularizer/CastCast5net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ´
&net_output/ActivityRegularizer/truedivRealDiv7net_output/ActivityRegularizer/PartitionedCall:output:0'net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
tf.math.subtract_11/Sub/yConst*
_output_shapes
:*
dtype0*Q
valueHBF"<ÍÌÌ=    ÍÌÌ=ÍÌÌ=ÍÌÌ=    ÍÌÌ=ÍÌÌ=ÍÌÌ="þ¾>ÍÌÌ=ÍÌÌ=&¬~>    ÍÌÌ=¡
tf.math.subtract_11/SubSub+net_output/StatefulPartitionedCall:output:0"tf.math.subtract_11/Sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
tf.math.pow_11/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
tf.math.pow_11/PowPowtf.math.subtract_11/Sub:z:0tf.math.pow_11/Pow/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
+tf.math.reduce_sum_11/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
tf.math.reduce_sum_11/SumSumtf.math.pow_11/Pow:z:04tf.math.reduce_sum_11/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
tf.math.reduce_mean_11/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
tf.math.reduce_mean_11/MeanMean"tf.math.reduce_sum_11/Sum:output:0%tf.math.reduce_mean_11/Const:output:0*
T0*
_output_shapes
: ^
tf.__operators__.add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *7
tf.__operators__.add_11/AddV2AddV2$tf.math.reduce_mean_11/Mean:output:0"tf.__operators__.add_11/y:output:0*
T0*
_output_shapes
: Ê
add_loss_11/PartitionedCallPartitionedCall!tf.__operators__.add_11/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_add_loss_11_layer_call_and_return_conditional_losses_61980m
IdentityIdentity"tf.math.reduce_sum_11/Sum:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿh

Identity_1Identity(dense_11/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: j

Identity_2Identity*net_output/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: d

Identity_3Identity$add_loss_11/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: 
NoOpNoOp!^dense_11/StatefulPartitionedCall#^net_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2H
"net_output/StatefulPartitionedCall"net_output/StatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_12
·
r
F__inference_add_loss_11_layer_call_and_return_conditional_losses_61980

inputs
identity

identity_1=
IdentityIdentityinputs*
T0*
_output_shapes
: ?

Identity_1Identityinputs*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
ê

«
G__inference_dense_11_layer_call_and_return_all_conditional_losses_62354

inputs
unknown:
identity

identity_1¢StatefulPartitionedCallË
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_61933£
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *8
f3R1
/__inference_dense_11_activity_regularizer_61905o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à
G
+__inference_add_loss_11_layer_call_fn_62376

inputs
identity£
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_add_loss_11_layer_call_and_return_conditional_losses_61980O
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
©:
¡
 __inference__wrapped_model_61892
input_12B
0model_29_dense_11_matmul_readvariableop_resource:D
2model_29_net_output_matmul_readvariableop_resource:
identity¢'model_29/dense_11/MatMul/ReadVariableOp¢)model_29/net_output/MatMul/ReadVariableOp
'model_29/dense_11/MatMul/ReadVariableOpReadVariableOp0model_29_dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
model_29/dense_11/MatMulMatMulinput_12/model_29/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
model_29/dense_11/ReluRelu"model_29/dense_11/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,model_29/dense_11/ActivityRegularizer/SquareSquare$model_29/dense_11/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
+model_29/dense_11/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¹
)model_29/dense_11/ActivityRegularizer/SumSum0model_29/dense_11/ActivityRegularizer/Square:y:04model_29/dense_11/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: p
+model_29/dense_11/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    »
)model_29/dense_11/ActivityRegularizer/mulMul4model_29/dense_11/ActivityRegularizer/mul/x:output:02model_29/dense_11/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 
+model_29/dense_11/ActivityRegularizer/ShapeShape$model_29/dense_11/Relu:activations:0*
T0*
_output_shapes
:
9model_29/dense_11/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
;model_29/dense_11/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;model_29/dense_11/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3model_29/dense_11/ActivityRegularizer/strided_sliceStridedSlice4model_29/dense_11/ActivityRegularizer/Shape:output:0Bmodel_29/dense_11/ActivityRegularizer/strided_slice/stack:output:0Dmodel_29/dense_11/ActivityRegularizer/strided_slice/stack_1:output:0Dmodel_29/dense_11/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask 
*model_29/dense_11/ActivityRegularizer/CastCast<model_29/dense_11/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ¸
-model_29/dense_11/ActivityRegularizer/truedivRealDiv-model_29/dense_11/ActivityRegularizer/mul:z:0.model_29/dense_11/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
)model_29/net_output/MatMul/ReadVariableOpReadVariableOp2model_29_net_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¯
model_29/net_output/MatMulMatMul$model_29/dense_11/Relu:activations:01model_29/net_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
model_29/net_output/ReluRelu$model_29/net_output/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.model_29/net_output/ActivityRegularizer/SquareSquare&model_29/net_output/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
-model_29/net_output/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¿
+model_29/net_output/ActivityRegularizer/SumSum2model_29/net_output/ActivityRegularizer/Square:y:06model_29/net_output/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: r
-model_29/net_output/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Á
+model_29/net_output/ActivityRegularizer/mulMul6model_29/net_output/ActivityRegularizer/mul/x:output:04model_29/net_output/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 
-model_29/net_output/ActivityRegularizer/ShapeShape&model_29/net_output/Relu:activations:0*
T0*
_output_shapes
:
;model_29/net_output/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=model_29/net_output/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=model_29/net_output/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5model_29/net_output/ActivityRegularizer/strided_sliceStridedSlice6model_29/net_output/ActivityRegularizer/Shape:output:0Dmodel_29/net_output/ActivityRegularizer/strided_slice/stack:output:0Fmodel_29/net_output/ActivityRegularizer/strided_slice/stack_1:output:0Fmodel_29/net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¤
,model_29/net_output/ActivityRegularizer/CastCast>model_29/net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ¾
/model_29/net_output/ActivityRegularizer/truedivRealDiv/model_29/net_output/ActivityRegularizer/mul:z:00model_29/net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: §
"model_29/tf.math.subtract_11/Sub/yConst*
_output_shapes
:*
dtype0*Q
valueHBF"<ÍÌÌ=    ÍÌÌ=ÍÌÌ=ÍÌÌ=    ÍÌÌ=ÍÌÌ=ÍÌÌ="þ¾>ÍÌÌ=ÍÌÌ=&¬~>    ÍÌÌ=®
 model_29/tf.math.subtract_11/SubSub&model_29/net_output/Relu:activations:0+model_29/tf.math.subtract_11/Sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
model_29/tf.math.pow_11/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @¢
model_29/tf.math.pow_11/PowPow$model_29/tf.math.subtract_11/Sub:z:0&model_29/tf.math.pow_11/Pow/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
4model_29/tf.math.reduce_sum_11/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ·
"model_29/tf.math.reduce_sum_11/SumSummodel_29/tf.math.pow_11/Pow:z:0=model_29/tf.math.reduce_sum_11/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
%model_29/tf.math.reduce_mean_11/ConstConst*
_output_shapes
:*
dtype0*
valueB: ª
$model_29/tf.math.reduce_mean_11/MeanMean+model_29/tf.math.reduce_sum_11/Sum:output:0.model_29/tf.math.reduce_mean_11/Const:output:0*
T0*
_output_shapes
: g
"model_29/tf.__operators__.add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *7¬
&model_29/tf.__operators__.add_11/AddV2AddV2-model_29/tf.math.reduce_mean_11/Mean:output:0+model_29/tf.__operators__.add_11/y:output:0*
T0*
_output_shapes
: v
IdentityIdentity+model_29/tf.math.reduce_sum_11/Sum:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp(^model_29/dense_11/MatMul/ReadVariableOp*^model_29/net_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2R
'model_29/dense_11/MatMul/ReadVariableOp'model_29/dense_11/MatMul/ReadVariableOp2V
)model_29/net_output/MatMul/ReadVariableOp)model_29/net_output/MatMul/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_12

F
/__inference_dense_11_activity_regularizer_61905
x
identity6
SquareSquarex*
T0*
_output_shapes
:9
RankRank
Square:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
í6
ß
!__inference__traced_restore_62508
file_prefix2
 assignvariableop_dense_11_kernel:6
$assignvariableop_1_net_output_kernel:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: "
assignvariableop_7_total: "
assignvariableop_8_count: ;
)assignvariableop_9_adam_dense_11_kernel_m:>
,assignvariableop_10_adam_net_output_kernel_m:<
*assignvariableop_11_adam_dense_11_kernel_v:>
,assignvariableop_12_adam_net_output_kernel_v:
identity_14¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ä
valueºB·B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B ä
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_dense_11_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp$assignvariableop_1_net_output_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_totalIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_countIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp)assignvariableop_9_adam_dense_11_kernel_mIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp,assignvariableop_10_adam_net_output_kernel_mIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp*assignvariableop_11_adam_dense_11_kernel_vIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp,assignvariableop_12_adam_net_output_kernel_vIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 í
Identity_13Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_14IdentityIdentity_13:output:0^NoOp_1*
T0*
_output_shapes
: Ú
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_14Identity_14:output:0*/
_input_shapes
: : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¶
serving_default¢
=
input_121
serving_default_input_12:0ÿÿÿÿÿÿÿÿÿE
tf.math.reduce_sum_11,
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¦X

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

	optimizer
loss
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
±

kernel
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
±

kernel
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
(
"	keras_api"
_tf_keras_layer
(
#	keras_api"
_tf_keras_layer
(
$	keras_api"
_tf_keras_layer
(
%	keras_api"
_tf_keras_layer
(
&	keras_api"
_tf_keras_layer
¥
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_layer
w
-iter

.beta_1

/beta_2
	0decay
1learning_ratemPmQvRvS"
	optimizer
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
î2ë
(__inference_model_29_layer_call_fn_61997
(__inference_model_29_layer_call_fn_62217
(__inference_model_29_layer_call_fn_62229
(__inference_model_29_layer_call_fn_62117À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ú2×
C__inference_model_29_layer_call_and_return_conditional_losses_62278
C__inference_model_29_layer_call_and_return_conditional_losses_62327
C__inference_model_29_layer_call_and_return_conditional_losses_62158
C__inference_model_29_layer_call_and_return_conditional_losses_62199À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÌBÉ
 __inference__wrapped_model_61892input_12"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
7serving_default"
signature_map
!:2dense_11/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
__call__
=activity_regularizer_fn
*&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_11_layer_call_fn_62345¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_11_layer_call_and_return_all_conditional_losses_62354¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
#:!2net_output/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
Dactivity_regularizer_fn
*!&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_net_output_layer_call_fn_62361¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_net_output_layer_call_and_return_all_conditional_losses_62370¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_add_loss_11_layer_call_fn_62376¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_add_loss_11_layer_call_and_return_conditional_losses_62381¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
'
K0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ËBÈ
#__inference_signature_wrapper_62338input_12"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
à2Ý
/__inference_dense_11_activity_regularizer_61905©
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢
	
í2ê
C__inference_dense_11_layer_call_and_return_conditional_losses_62389¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
â2ß
1__inference_net_output_activity_regularizer_61918©
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢
	
ï2ì
E__inference_net_output_layer_call_and_return_conditional_losses_62397¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Ltotal
	Mcount
N	variables
O	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
L0
M1"
trackable_list_wrapper
-
N	variables"
_generic_user_object
&:$2Adam/dense_11/kernel/m
(:&2Adam/net_output/kernel/m
&:$2Adam/dense_11/kernel/v
(:&2Adam/net_output/kernel/v§
 __inference__wrapped_model_618921¢.
'¢$
"
input_12ÿÿÿÿÿÿÿÿÿ
ª "IªF
D
tf.math.reduce_sum_11+(
tf.math.reduce_sum_11ÿÿÿÿÿÿÿÿÿ
F__inference_add_loss_11_layer_call_and_return_conditional_losses_62381D¢
¢

inputs 
ª ""¢


0 

	
1/0 X
+__inference_add_loss_11_layer_call_fn_62376)¢
¢

inputs 
ª " Y
/__inference_dense_11_activity_regularizer_61905&¢
¢
	
x
ª " ´
G__inference_dense_11_layer_call_and_return_all_conditional_losses_62354i/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ

	
1/0 ¢
C__inference_dense_11_layer_call_and_return_conditional_losses_62389[/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
(__inference_dense_11_layer_call_fn_62345N/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÔ
C__inference_model_29_layer_call_and_return_conditional_losses_621589¢6
/¢,
"
input_12ÿÿÿÿÿÿÿÿÿ
p 

 
ª "K¢H

0ÿÿÿÿÿÿÿÿÿ
-*
	
1/0 
	
1/1 
	
1/2 Ô
C__inference_model_29_layer_call_and_return_conditional_losses_621999¢6
/¢,
"
input_12ÿÿÿÿÿÿÿÿÿ
p

 
ª "K¢H

0ÿÿÿÿÿÿÿÿÿ
-*
	
1/0 
	
1/1 
	
1/2 Ò
C__inference_model_29_layer_call_and_return_conditional_losses_622787¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "K¢H

0ÿÿÿÿÿÿÿÿÿ
-*
	
1/0 
	
1/1 
	
1/2 Ò
C__inference_model_29_layer_call_and_return_conditional_losses_623277¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "K¢H

0ÿÿÿÿÿÿÿÿÿ
-*
	
1/0 
	
1/1 
	
1/2 
(__inference_model_29_layer_call_fn_61997U9¢6
/¢,
"
input_12ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_model_29_layer_call_fn_62117U9¢6
/¢,
"
input_12ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_model_29_layer_call_fn_62217S7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_model_29_layer_call_fn_62229S7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ[
1__inference_net_output_activity_regularizer_61918&¢
¢
	
x
ª " ¶
I__inference_net_output_layer_call_and_return_all_conditional_losses_62370i/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ

	
1/0 ¤
E__inference_net_output_layer_call_and_return_conditional_losses_62397[/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
*__inference_net_output_layer_call_fn_62361N/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¶
#__inference_signature_wrapper_62338=¢:
¢ 
3ª0
.
input_12"
input_12ÿÿÿÿÿÿÿÿÿ"IªF
D
tf.math.reduce_sum_11+(
tf.math.reduce_sum_11ÿÿÿÿÿÿÿÿÿ