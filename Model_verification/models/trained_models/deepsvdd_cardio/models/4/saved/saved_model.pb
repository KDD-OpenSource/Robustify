┌ї
Ёо
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
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
Ї
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
delete_old_dirsbool(ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ
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
dtypetypeѕ
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
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
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
┴
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
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
Ш
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
ї
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Лю
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
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
є
Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_9/kernel/m

)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes

:*
dtype0
ї
Adam/net_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/net_output/kernel/m
Ё
,Adam/net_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/net_output/kernel/m*
_output_shapes

:*
dtype0
є
Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_9/kernel/v

)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes

:*
dtype0
ї
Adam/net_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/net_output/kernel/v
Ё
,Adam/net_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/net_output/kernel/v*
_output_shapes

:*
dtype0

NoOpNoOp
Б 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*я
valueнBЛ B╩
 
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
ю

kernel
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
ю

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
ј
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
░
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
^X
VARIABLE_VALUEdense_9/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 
░
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
░
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
Љ
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
Ђ{
VARIABLE_VALUEAdam/dense_9/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ё~
VARIABLE_VALUEAdam/net_output/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Ђ{
VARIABLE_VALUEAdam/dense_9/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ё~
VARIABLE_VALUEAdam/net_output/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
serving_default_input_10Placeholder*'
_output_shapes
:         *
dtype0*
shape:         
Н
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_10dense_9/kernelnet_output/kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *,
f'R%
#__inference_signature_wrapper_51791
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ѓ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_9/kernel/Read/ReadVariableOp%net_output/kernel/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp,Adam/net_output/kernel/m/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp,Adam/net_output/kernel/v/Read/ReadVariableOpConst*
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
GPU 2J 8ѓ *'
f"R 
__inference__traced_save_51912
Щ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_9/kernelnet_output/kernel	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_9/kernel/mAdam/net_output/kernel/mAdam/dense_9/kernel/vAdam/net_output/kernel/v*
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
GPU 2J 8ѓ **
f%R#
!__inference__traced_restore_51961фо
Х
q
E__inference_add_loss_9_layer_call_and_return_conditional_losses_51433

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
є7
╠
C__inference_model_24_layer_call_and_return_conditional_losses_51731

inputs8
&dense_9_matmul_readvariableop_resource:;
)net_output_matmul_readvariableop_resource:
identity

identity_1

identity_2

identity_3ѕбdense_9/MatMul/ReadVariableOpб net_output/MatMul/ReadVariableOpё
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype0y
dense_9/MatMulMatMulinputs%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `
dense_9/ReluReludense_9/MatMul:product:0*
T0*'
_output_shapes
:         z
"dense_9/ActivityRegularizer/SquareSquaredense_9/Relu:activations:0*
T0*'
_output_shapes
:         r
!dense_9/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Џ
dense_9/ActivityRegularizer/SumSum&dense_9/ActivityRegularizer/Square:y:0*dense_9/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_9/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ю
dense_9/ActivityRegularizer/mulMul*dense_9/ActivityRegularizer/mul/x:output:0(dense_9/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: k
!dense_9/ActivityRegularizer/ShapeShapedense_9/Relu:activations:0*
T0*
_output_shapes
:y
/dense_9/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_9/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_9/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
)dense_9/ActivityRegularizer/strided_sliceStridedSlice*dense_9/ActivityRegularizer/Shape:output:08dense_9/ActivityRegularizer/strided_slice/stack:output:0:dense_9/ActivityRegularizer/strided_slice/stack_1:output:0:dense_9/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskї
 dense_9/ActivityRegularizer/CastCast2dense_9/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: џ
#dense_9/ActivityRegularizer/truedivRealDiv#dense_9/ActivityRegularizer/mul:z:0$dense_9/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: і
 net_output/MatMul/ReadVariableOpReadVariableOp)net_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Њ
net_output/MatMulMatMuldense_9/Relu:activations:0(net_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         f
net_output/ReluRelunet_output/MatMul:product:0*
T0*'
_output_shapes
:         ђ
%net_output/ActivityRegularizer/SquareSquarenet_output/Relu:activations:0*
T0*'
_output_shapes
:         u
$net_output/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ц
"net_output/ActivityRegularizer/SumSum)net_output/ActivityRegularizer/Square:y:0-net_output/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: i
$net_output/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    д
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
valueB:В
,net_output/ActivityRegularizer/strided_sliceStridedSlice-net_output/ActivityRegularizer/Shape:output:0;net_output/ActivityRegularizer/strided_slice/stack:output:0=net_output/ActivityRegularizer/strided_slice/stack_1:output:0=net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskњ
#net_output/ActivityRegularizer/CastCast5net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Б
&net_output/ActivityRegularizer/truedivRealDiv&net_output/ActivityRegularizer/mul:z:0'net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: Ю
tf.math.subtract_9/Sub/yConst*
_output_shapes
:*
dtype0*Q
valueHBF"<    ═╠╠=Ўnя>═╠╠=*ё8>═╠╠=Ъ=>═╠╠=╝ д>═╠╠=═╠╠=═╠╠=═╠╠=    ═╠╠=Љ
tf.math.subtract_9/SubSubnet_output/Relu:activations:0!tf.math.subtract_9/Sub/y:output:0*
T0*'
_output_shapes
:         X
tf.math.pow_9/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @ё
tf.math.pow_9/PowPowtf.math.subtract_9/Sub:z:0tf.math.pow_9/Pow/y:output:0*
T0*'
_output_shapes
:         u
*tf.math.reduce_sum_9/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         Ў
tf.math.reduce_sum_9/SumSumtf.math.pow_9/Pow:z:03tf.math.reduce_sum_9/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         e
tf.math.reduce_mean_9/ConstConst*
_output_shapes
:*
dtype0*
valueB: ї
tf.math.reduce_mean_9/MeanMean!tf.math.reduce_sum_9/Sum:output:0$tf.math.reduce_mean_9/Const:output:0*
T0*
_output_shapes
: ]
tf.__operators__.add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *P7ј
tf.__operators__.add_9/AddV2AddV2#tf.math.reduce_mean_9/Mean:output:0!tf.__operators__.add_9/y:output:0*
T0*
_output_shapes
: l
IdentityIdentity!tf.math.reduce_sum_9/Sum:output:0^NoOp*
T0*#
_output_shapes
:         g

Identity_1Identity'dense_9/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: j

Identity_2Identity*net_output/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: `

Identity_3Identity tf.__operators__.add_9/AddV2:z:0^NoOp*
T0*
_output_shapes
: Ѕ
NoOpNoOp^dense_9/MatMul/ReadVariableOp!^net_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2D
 net_output/MatMul/ReadVariableOp net_output/MatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┼
Ў
(__inference_model_24_layer_call_fn_51682

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         : : : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_model_24_layer_call_and_return_conditional_losses_51548k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Є
E
.__inference_dense_9_activity_regularizer_51358
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
:         G
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
у

ф
F__inference_dense_9_layer_call_and_return_all_conditional_losses_51807

inputs
unknown:
identity

identity_1ѕбStatefulPartitionedCall╩
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_51386б
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
GPU 2J 8ѓ *7
f2R0
.__inference_dense_9_activity_regularizer_51358o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         X

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
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
­9
Ъ
 __inference__wrapped_model_51345
input_10A
/model_24_dense_9_matmul_readvariableop_resource:D
2model_24_net_output_matmul_readvariableop_resource:
identityѕб&model_24/dense_9/MatMul/ReadVariableOpб)model_24/net_output/MatMul/ReadVariableOpќ
&model_24/dense_9/MatMul/ReadVariableOpReadVariableOp/model_24_dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ї
model_24/dense_9/MatMulMatMulinput_10.model_24/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
model_24/dense_9/ReluRelu!model_24/dense_9/MatMul:product:0*
T0*'
_output_shapes
:         ї
+model_24/dense_9/ActivityRegularizer/SquareSquare#model_24/dense_9/Relu:activations:0*
T0*'
_output_shapes
:         {
*model_24/dense_9/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Х
(model_24/dense_9/ActivityRegularizer/SumSum/model_24/dense_9/ActivityRegularizer/Square:y:03model_24/dense_9/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: o
*model_24/dense_9/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    И
(model_24/dense_9/ActivityRegularizer/mulMul3model_24/dense_9/ActivityRegularizer/mul/x:output:01model_24/dense_9/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: }
*model_24/dense_9/ActivityRegularizer/ShapeShape#model_24/dense_9/Relu:activations:0*
T0*
_output_shapes
:ѓ
8model_24/dense_9/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ё
:model_24/dense_9/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ё
:model_24/dense_9/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:і
2model_24/dense_9/ActivityRegularizer/strided_sliceStridedSlice3model_24/dense_9/ActivityRegularizer/Shape:output:0Amodel_24/dense_9/ActivityRegularizer/strided_slice/stack:output:0Cmodel_24/dense_9/ActivityRegularizer/strided_slice/stack_1:output:0Cmodel_24/dense_9/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskъ
)model_24/dense_9/ActivityRegularizer/CastCast;model_24/dense_9/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: х
,model_24/dense_9/ActivityRegularizer/truedivRealDiv,model_24/dense_9/ActivityRegularizer/mul:z:0-model_24/dense_9/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ю
)model_24/net_output/MatMul/ReadVariableOpReadVariableOp2model_24_net_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0«
model_24/net_output/MatMulMatMul#model_24/dense_9/Relu:activations:01model_24/net_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x
model_24/net_output/ReluRelu$model_24/net_output/MatMul:product:0*
T0*'
_output_shapes
:         њ
.model_24/net_output/ActivityRegularizer/SquareSquare&model_24/net_output/Relu:activations:0*
T0*'
_output_shapes
:         ~
-model_24/net_output/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ┐
+model_24/net_output/ActivityRegularizer/SumSum2model_24/net_output/ActivityRegularizer/Square:y:06model_24/net_output/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: r
-model_24/net_output/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ┴
+model_24/net_output/ActivityRegularizer/mulMul6model_24/net_output/ActivityRegularizer/mul/x:output:04model_24/net_output/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: Ѓ
-model_24/net_output/ActivityRegularizer/ShapeShape&model_24/net_output/Relu:activations:0*
T0*
_output_shapes
:Ё
;model_24/net_output/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Є
=model_24/net_output/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Є
=model_24/net_output/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
5model_24/net_output/ActivityRegularizer/strided_sliceStridedSlice6model_24/net_output/ActivityRegularizer/Shape:output:0Dmodel_24/net_output/ActivityRegularizer/strided_slice/stack:output:0Fmodel_24/net_output/ActivityRegularizer/strided_slice/stack_1:output:0Fmodel_24/net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskц
,model_24/net_output/ActivityRegularizer/CastCast>model_24/net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Й
/model_24/net_output/ActivityRegularizer/truedivRealDiv/model_24/net_output/ActivityRegularizer/mul:z:00model_24/net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: д
!model_24/tf.math.subtract_9/Sub/yConst*
_output_shapes
:*
dtype0*Q
valueHBF"<    ═╠╠=Ўnя>═╠╠=*ё8>═╠╠=Ъ=>═╠╠=╝ д>═╠╠=═╠╠=═╠╠=═╠╠=    ═╠╠=г
model_24/tf.math.subtract_9/SubSub&model_24/net_output/Relu:activations:0*model_24/tf.math.subtract_9/Sub/y:output:0*
T0*'
_output_shapes
:         a
model_24/tf.math.pow_9/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Ъ
model_24/tf.math.pow_9/PowPow#model_24/tf.math.subtract_9/Sub:z:0%model_24/tf.math.pow_9/Pow/y:output:0*
T0*'
_output_shapes
:         ~
3model_24/tf.math.reduce_sum_9/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
!model_24/tf.math.reduce_sum_9/SumSummodel_24/tf.math.pow_9/Pow:z:0<model_24/tf.math.reduce_sum_9/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         n
$model_24/tf.math.reduce_mean_9/ConstConst*
_output_shapes
:*
dtype0*
valueB: Д
#model_24/tf.math.reduce_mean_9/MeanMean*model_24/tf.math.reduce_sum_9/Sum:output:0-model_24/tf.math.reduce_mean_9/Const:output:0*
T0*
_output_shapes
: f
!model_24/tf.__operators__.add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *P7Е
%model_24/tf.__operators__.add_9/AddV2AddV2,model_24/tf.math.reduce_mean_9/Mean:output:0*model_24/tf.__operators__.add_9/y:output:0*
T0*
_output_shapes
: u
IdentityIdentity*model_24/tf.math.reduce_sum_9/Sum:output:0^NoOp*
T0*#
_output_shapes
:         Џ
NoOpNoOp'^model_24/dense_9/MatMul/ReadVariableOp*^model_24/net_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 2P
&model_24/dense_9/MatMul/ReadVariableOp&model_24/dense_9/MatMul/ReadVariableOp2V
)model_24/net_output/MatMul/ReadVariableOp)model_24/net_output/MatMul/ReadVariableOp:Q M
'
_output_shapes
:         
"
_user_specified_name
input_10
у6
▄
!__inference__traced_restore_51961
file_prefix1
assignvariableop_dense_9_kernel:6
$assignvariableop_1_net_output_kernel:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: "
assignvariableop_7_total: "
assignvariableop_8_count: :
(assignvariableop_9_adam_dense_9_kernel_m:>
,assignvariableop_10_adam_net_output_kernel_m:;
)assignvariableop_11_adam_dense_9_kernel_v:>
,assignvariableop_12_adam_net_output_kernel_v:
identity_14ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_2бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9ъ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*─
value║BиB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHї
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B С
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOpAssignVariableOpassignvariableop_dense_9_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_1AssignVariableOp$assignvariableop_1_net_output_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:І
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:ћ
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_7AssignVariableOpassignvariableop_7_totalIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_8AssignVariableOpassignvariableop_8_countIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_9AssignVariableOp(assignvariableop_9_adam_dense_9_kernel_mIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_10AssignVariableOp,assignvariableop_10_adam_net_output_kernel_mIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_11AssignVariableOp)assignvariableop_11_adam_dense_9_kernel_vIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_12AssignVariableOp,assignvariableop_12_adam_net_output_kernel_vIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ь
Identity_13Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_14IdentityIdentity_13:output:0^NoOp_1*
T0*
_output_shapes
: ┌
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
_user_specified_namefile_prefix
ў
Ф
B__inference_dense_9_layer_call_and_return_conditional_losses_51386

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         ^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ў
Ф
B__inference_dense_9_layer_call_and_return_conditional_losses_51842

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         ^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Х
q
E__inference_add_loss_9_layer_call_and_return_conditional_losses_51834

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
┼
Ў
(__inference_model_24_layer_call_fn_51670

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         : : : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_model_24_layer_call_and_return_conditional_losses_51440k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┼4
а
C__inference_model_24_layer_call_and_return_conditional_losses_51652
input_10
dense_9_51614:"
net_output_51625:
identity

identity_1

identity_2

identity_3ѕбdense_9/StatefulPartitionedCallб"net_output/StatefulPartitionedCall┌
dense_9/StatefulPartitionedCallStatefulPartitionedCallinput_10dense_9_51614*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_51386к
+dense_9/ActivityRegularizer/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
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
GPU 2J 8ѓ *7
f2R0
.__inference_dense_9_activity_regularizer_51358y
!dense_9/ActivityRegularizer/ShapeShape(dense_9/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:y
/dense_9/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_9/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_9/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
)dense_9/ActivityRegularizer/strided_sliceStridedSlice*dense_9/ActivityRegularizer/Shape:output:08dense_9/ActivityRegularizer/strided_slice/stack:output:0:dense_9/ActivityRegularizer/strided_slice/stack_1:output:0:dense_9/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskї
 dense_9/ActivityRegularizer/CastCast2dense_9/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ф
#dense_9/ActivityRegularizer/truedivRealDiv4dense_9/ActivityRegularizer/PartitionedCall:output:0$dense_9/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: Ѓ
"net_output/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0net_output_51625*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_net_output_layer_call_and_return_conditional_losses_51406¤
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
GPU 2J 8ѓ *:
f5R3
1__inference_net_output_activity_regularizer_51371
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
valueB:В
,net_output/ActivityRegularizer/strided_sliceStridedSlice-net_output/ActivityRegularizer/Shape:output:0;net_output/ActivityRegularizer/strided_slice/stack:output:0=net_output/ActivityRegularizer/strided_slice/stack_1:output:0=net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskњ
#net_output/ActivityRegularizer/CastCast5net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ┤
&net_output/ActivityRegularizer/truedivRealDiv7net_output/ActivityRegularizer/PartitionedCall:output:0'net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: Ю
tf.math.subtract_9/Sub/yConst*
_output_shapes
:*
dtype0*Q
valueHBF"<    ═╠╠=Ўnя>═╠╠=*ё8>═╠╠=Ъ=>═╠╠=╝ д>═╠╠=═╠╠=═╠╠=═╠╠=    ═╠╠=Ъ
tf.math.subtract_9/SubSub+net_output/StatefulPartitionedCall:output:0!tf.math.subtract_9/Sub/y:output:0*
T0*'
_output_shapes
:         X
tf.math.pow_9/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @ё
tf.math.pow_9/PowPowtf.math.subtract_9/Sub:z:0tf.math.pow_9/Pow/y:output:0*
T0*'
_output_shapes
:         u
*tf.math.reduce_sum_9/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         Ў
tf.math.reduce_sum_9/SumSumtf.math.pow_9/Pow:z:03tf.math.reduce_sum_9/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         e
tf.math.reduce_mean_9/ConstConst*
_output_shapes
:*
dtype0*
valueB: ї
tf.math.reduce_mean_9/MeanMean!tf.math.reduce_sum_9/Sum:output:0$tf.math.reduce_mean_9/Const:output:0*
T0*
_output_shapes
: ]
tf.__operators__.add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *P7ј
tf.__operators__.add_9/AddV2AddV2#tf.math.reduce_mean_9/Mean:output:0!tf.__operators__.add_9/y:output:0*
T0*
_output_shapes
: К
add_loss_9/PartitionedCallPartitionedCall tf.__operators__.add_9/AddV2:z:0*
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
GPU 2J 8ѓ *N
fIRG
E__inference_add_loss_9_layer_call_and_return_conditional_losses_51433l
IdentityIdentity!tf.math.reduce_sum_9/Sum:output:0^NoOp*
T0*#
_output_shapes
:         g

Identity_1Identity'dense_9/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: j

Identity_2Identity*net_output/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: c

Identity_3Identity#add_loss_9/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: Ї
NoOpNoOp ^dense_9/StatefulPartitionedCall#^net_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2H
"net_output/StatefulPartitionedCall"net_output/StatefulPartitionedCall:Q M
'
_output_shapes
:         
"
_user_specified_name
input_10
В$
о
__inference__traced_save_51912
file_prefix-
)savev2_dense_9_kernel_read_readvariableop0
,savev2_net_output_kernel_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop7
3savev2_adam_net_output_kernel_m_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop7
3savev2_adam_net_output_kernel_v_read_readvariableop
savev2_const

identity_1ѕбMergeV2Checkpointsw
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
_temp/partЂ
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
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Џ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*─
value║BиB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЅ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B З
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_9_kernel_read_readvariableop,savev2_net_output_kernel_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop3savev2_adam_net_output_kernel_m_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop3savev2_adam_net_output_kernel_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:І
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
џ
ќ
#__inference_signature_wrapper_51791
input_10
unknown:
	unknown_0:
identityѕбStatefulPartitionedCall│
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *)
f$R"
 __inference__wrapped_model_51345k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         
"
_user_specified_name
input_10
є7
╠
C__inference_model_24_layer_call_and_return_conditional_losses_51780

inputs8
&dense_9_matmul_readvariableop_resource:;
)net_output_matmul_readvariableop_resource:
identity

identity_1

identity_2

identity_3ѕбdense_9/MatMul/ReadVariableOpб net_output/MatMul/ReadVariableOpё
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype0y
dense_9/MatMulMatMulinputs%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         `
dense_9/ReluReludense_9/MatMul:product:0*
T0*'
_output_shapes
:         z
"dense_9/ActivityRegularizer/SquareSquaredense_9/Relu:activations:0*
T0*'
_output_shapes
:         r
!dense_9/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Џ
dense_9/ActivityRegularizer/SumSum&dense_9/ActivityRegularizer/Square:y:0*dense_9/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_9/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ю
dense_9/ActivityRegularizer/mulMul*dense_9/ActivityRegularizer/mul/x:output:0(dense_9/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: k
!dense_9/ActivityRegularizer/ShapeShapedense_9/Relu:activations:0*
T0*
_output_shapes
:y
/dense_9/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_9/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_9/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
)dense_9/ActivityRegularizer/strided_sliceStridedSlice*dense_9/ActivityRegularizer/Shape:output:08dense_9/ActivityRegularizer/strided_slice/stack:output:0:dense_9/ActivityRegularizer/strided_slice/stack_1:output:0:dense_9/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskї
 dense_9/ActivityRegularizer/CastCast2dense_9/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: џ
#dense_9/ActivityRegularizer/truedivRealDiv#dense_9/ActivityRegularizer/mul:z:0$dense_9/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: і
 net_output/MatMul/ReadVariableOpReadVariableOp)net_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Њ
net_output/MatMulMatMuldense_9/Relu:activations:0(net_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         f
net_output/ReluRelunet_output/MatMul:product:0*
T0*'
_output_shapes
:         ђ
%net_output/ActivityRegularizer/SquareSquarenet_output/Relu:activations:0*
T0*'
_output_shapes
:         u
$net_output/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ц
"net_output/ActivityRegularizer/SumSum)net_output/ActivityRegularizer/Square:y:0-net_output/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: i
$net_output/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    д
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
valueB:В
,net_output/ActivityRegularizer/strided_sliceStridedSlice-net_output/ActivityRegularizer/Shape:output:0;net_output/ActivityRegularizer/strided_slice/stack:output:0=net_output/ActivityRegularizer/strided_slice/stack_1:output:0=net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskњ
#net_output/ActivityRegularizer/CastCast5net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Б
&net_output/ActivityRegularizer/truedivRealDiv&net_output/ActivityRegularizer/mul:z:0'net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: Ю
tf.math.subtract_9/Sub/yConst*
_output_shapes
:*
dtype0*Q
valueHBF"<    ═╠╠=Ўnя>═╠╠=*ё8>═╠╠=Ъ=>═╠╠=╝ д>═╠╠=═╠╠=═╠╠=═╠╠=    ═╠╠=Љ
tf.math.subtract_9/SubSubnet_output/Relu:activations:0!tf.math.subtract_9/Sub/y:output:0*
T0*'
_output_shapes
:         X
tf.math.pow_9/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @ё
tf.math.pow_9/PowPowtf.math.subtract_9/Sub:z:0tf.math.pow_9/Pow/y:output:0*
T0*'
_output_shapes
:         u
*tf.math.reduce_sum_9/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         Ў
tf.math.reduce_sum_9/SumSumtf.math.pow_9/Pow:z:03tf.math.reduce_sum_9/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         e
tf.math.reduce_mean_9/ConstConst*
_output_shapes
:*
dtype0*
valueB: ї
tf.math.reduce_mean_9/MeanMean!tf.math.reduce_sum_9/Sum:output:0$tf.math.reduce_mean_9/Const:output:0*
T0*
_output_shapes
: ]
tf.__operators__.add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *P7ј
tf.__operators__.add_9/AddV2AddV2#tf.math.reduce_mean_9/Mean:output:0!tf.__operators__.add_9/y:output:0*
T0*
_output_shapes
: l
IdentityIdentity!tf.math.reduce_sum_9/Sum:output:0^NoOp*
T0*#
_output_shapes
:         g

Identity_1Identity'dense_9/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: j

Identity_2Identity*net_output/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: `

Identity_3Identity tf.__operators__.add_9/AddV2:z:0^NoOp*
T0*
_output_shapes
: Ѕ
NoOpNoOp^dense_9/MatMul/ReadVariableOp!^net_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2D
 net_output/MatMul/ReadVariableOp net_output/MatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┼4
а
C__inference_model_24_layer_call_and_return_conditional_losses_51611
input_10
dense_9_51573:"
net_output_51584:
identity

identity_1

identity_2

identity_3ѕбdense_9/StatefulPartitionedCallб"net_output/StatefulPartitionedCall┌
dense_9/StatefulPartitionedCallStatefulPartitionedCallinput_10dense_9_51573*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_51386к
+dense_9/ActivityRegularizer/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
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
GPU 2J 8ѓ *7
f2R0
.__inference_dense_9_activity_regularizer_51358y
!dense_9/ActivityRegularizer/ShapeShape(dense_9/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:y
/dense_9/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_9/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_9/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
)dense_9/ActivityRegularizer/strided_sliceStridedSlice*dense_9/ActivityRegularizer/Shape:output:08dense_9/ActivityRegularizer/strided_slice/stack:output:0:dense_9/ActivityRegularizer/strided_slice/stack_1:output:0:dense_9/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskї
 dense_9/ActivityRegularizer/CastCast2dense_9/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ф
#dense_9/ActivityRegularizer/truedivRealDiv4dense_9/ActivityRegularizer/PartitionedCall:output:0$dense_9/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: Ѓ
"net_output/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0net_output_51584*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_net_output_layer_call_and_return_conditional_losses_51406¤
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
GPU 2J 8ѓ *:
f5R3
1__inference_net_output_activity_regularizer_51371
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
valueB:В
,net_output/ActivityRegularizer/strided_sliceStridedSlice-net_output/ActivityRegularizer/Shape:output:0;net_output/ActivityRegularizer/strided_slice/stack:output:0=net_output/ActivityRegularizer/strided_slice/stack_1:output:0=net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskњ
#net_output/ActivityRegularizer/CastCast5net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ┤
&net_output/ActivityRegularizer/truedivRealDiv7net_output/ActivityRegularizer/PartitionedCall:output:0'net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: Ю
tf.math.subtract_9/Sub/yConst*
_output_shapes
:*
dtype0*Q
valueHBF"<    ═╠╠=Ўnя>═╠╠=*ё8>═╠╠=Ъ=>═╠╠=╝ д>═╠╠=═╠╠=═╠╠=═╠╠=    ═╠╠=Ъ
tf.math.subtract_9/SubSub+net_output/StatefulPartitionedCall:output:0!tf.math.subtract_9/Sub/y:output:0*
T0*'
_output_shapes
:         X
tf.math.pow_9/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @ё
tf.math.pow_9/PowPowtf.math.subtract_9/Sub:z:0tf.math.pow_9/Pow/y:output:0*
T0*'
_output_shapes
:         u
*tf.math.reduce_sum_9/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         Ў
tf.math.reduce_sum_9/SumSumtf.math.pow_9/Pow:z:03tf.math.reduce_sum_9/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         e
tf.math.reduce_mean_9/ConstConst*
_output_shapes
:*
dtype0*
valueB: ї
tf.math.reduce_mean_9/MeanMean!tf.math.reduce_sum_9/Sum:output:0$tf.math.reduce_mean_9/Const:output:0*
T0*
_output_shapes
: ]
tf.__operators__.add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *P7ј
tf.__operators__.add_9/AddV2AddV2#tf.math.reduce_mean_9/Mean:output:0!tf.__operators__.add_9/y:output:0*
T0*
_output_shapes
: К
add_loss_9/PartitionedCallPartitionedCall tf.__operators__.add_9/AddV2:z:0*
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
GPU 2J 8ѓ *N
fIRG
E__inference_add_loss_9_layer_call_and_return_conditional_losses_51433l
IdentityIdentity!tf.math.reduce_sum_9/Sum:output:0^NoOp*
T0*#
_output_shapes
:         g

Identity_1Identity'dense_9/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: j

Identity_2Identity*net_output/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: c

Identity_3Identity#add_loss_9/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: Ї
NoOpNoOp ^dense_9/StatefulPartitionedCall#^net_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2H
"net_output/StatefulPartitionedCall"net_output/StatefulPartitionedCall:Q M
'
_output_shapes
:         
"
_user_specified_name
input_10
┐4
ъ
C__inference_model_24_layer_call_and_return_conditional_losses_51548

inputs
dense_9_51510:"
net_output_51521:
identity

identity_1

identity_2

identity_3ѕбdense_9/StatefulPartitionedCallб"net_output/StatefulPartitionedCallп
dense_9/StatefulPartitionedCallStatefulPartitionedCallinputsdense_9_51510*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_51386к
+dense_9/ActivityRegularizer/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
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
GPU 2J 8ѓ *7
f2R0
.__inference_dense_9_activity_regularizer_51358y
!dense_9/ActivityRegularizer/ShapeShape(dense_9/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:y
/dense_9/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_9/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_9/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
)dense_9/ActivityRegularizer/strided_sliceStridedSlice*dense_9/ActivityRegularizer/Shape:output:08dense_9/ActivityRegularizer/strided_slice/stack:output:0:dense_9/ActivityRegularizer/strided_slice/stack_1:output:0:dense_9/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskї
 dense_9/ActivityRegularizer/CastCast2dense_9/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ф
#dense_9/ActivityRegularizer/truedivRealDiv4dense_9/ActivityRegularizer/PartitionedCall:output:0$dense_9/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: Ѓ
"net_output/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0net_output_51521*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_net_output_layer_call_and_return_conditional_losses_51406¤
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
GPU 2J 8ѓ *:
f5R3
1__inference_net_output_activity_regularizer_51371
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
valueB:В
,net_output/ActivityRegularizer/strided_sliceStridedSlice-net_output/ActivityRegularizer/Shape:output:0;net_output/ActivityRegularizer/strided_slice/stack:output:0=net_output/ActivityRegularizer/strided_slice/stack_1:output:0=net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskњ
#net_output/ActivityRegularizer/CastCast5net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ┤
&net_output/ActivityRegularizer/truedivRealDiv7net_output/ActivityRegularizer/PartitionedCall:output:0'net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: Ю
tf.math.subtract_9/Sub/yConst*
_output_shapes
:*
dtype0*Q
valueHBF"<    ═╠╠=Ўnя>═╠╠=*ё8>═╠╠=Ъ=>═╠╠=╝ д>═╠╠=═╠╠=═╠╠=═╠╠=    ═╠╠=Ъ
tf.math.subtract_9/SubSub+net_output/StatefulPartitionedCall:output:0!tf.math.subtract_9/Sub/y:output:0*
T0*'
_output_shapes
:         X
tf.math.pow_9/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @ё
tf.math.pow_9/PowPowtf.math.subtract_9/Sub:z:0tf.math.pow_9/Pow/y:output:0*
T0*'
_output_shapes
:         u
*tf.math.reduce_sum_9/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         Ў
tf.math.reduce_sum_9/SumSumtf.math.pow_9/Pow:z:03tf.math.reduce_sum_9/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         e
tf.math.reduce_mean_9/ConstConst*
_output_shapes
:*
dtype0*
valueB: ї
tf.math.reduce_mean_9/MeanMean!tf.math.reduce_sum_9/Sum:output:0$tf.math.reduce_mean_9/Const:output:0*
T0*
_output_shapes
: ]
tf.__operators__.add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *P7ј
tf.__operators__.add_9/AddV2AddV2#tf.math.reduce_mean_9/Mean:output:0!tf.__operators__.add_9/y:output:0*
T0*
_output_shapes
: К
add_loss_9/PartitionedCallPartitionedCall tf.__operators__.add_9/AddV2:z:0*
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
GPU 2J 8ѓ *N
fIRG
E__inference_add_loss_9_layer_call_and_return_conditional_losses_51433l
IdentityIdentity!tf.math.reduce_sum_9/Sum:output:0^NoOp*
T0*#
_output_shapes
:         g

Identity_1Identity'dense_9/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: j

Identity_2Identity*net_output/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: c

Identity_3Identity#add_loss_9/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: Ї
NoOpNoOp ^dense_9/StatefulPartitionedCall#^net_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2H
"net_output/StatefulPartitionedCall"net_output/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Џ
«
E__inference_net_output_layer_call_and_return_conditional_losses_51850

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         ^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
­

Г
I__inference_net_output_layer_call_and_return_all_conditional_losses_51823

inputs
unknown:
identity

identity_1ѕбStatefulPartitionedCall═
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_net_output_layer_call_and_return_conditional_losses_51406Ц
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
GPU 2J 8ѓ *:
f5R3
1__inference_net_output_activity_regularizer_51371o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         X

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
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╦
Џ
(__inference_model_24_layer_call_fn_51570
input_10
unknown:
	unknown_0:
identityѕбStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         : : : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_model_24_layer_call_and_return_conditional_losses_51548k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         
"
_user_specified_name
input_10
╦
Џ
(__inference_model_24_layer_call_fn_51450
input_10
unknown:
	unknown_0:
identityѕбStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         : : : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_model_24_layer_call_and_return_conditional_losses_51440k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         
"
_user_specified_name
input_10
Џ
«
E__inference_net_output_layer_call_and_return_conditional_losses_51406

inputs0
matmul_readvariableop_resource:
identityѕбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         P
ReluReluMatMul:product:0*
T0*'
_output_shapes
:         a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         ^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┐4
ъ
C__inference_model_24_layer_call_and_return_conditional_losses_51440

inputs
dense_9_51387:"
net_output_51407:
identity

identity_1

identity_2

identity_3ѕбdense_9/StatefulPartitionedCallб"net_output/StatefulPartitionedCallп
dense_9/StatefulPartitionedCallStatefulPartitionedCallinputsdense_9_51387*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_51386к
+dense_9/ActivityRegularizer/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
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
GPU 2J 8ѓ *7
f2R0
.__inference_dense_9_activity_regularizer_51358y
!dense_9/ActivityRegularizer/ShapeShape(dense_9/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:y
/dense_9/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_9/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_9/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
)dense_9/ActivityRegularizer/strided_sliceStridedSlice*dense_9/ActivityRegularizer/Shape:output:08dense_9/ActivityRegularizer/strided_slice/stack:output:0:dense_9/ActivityRegularizer/strided_slice/stack_1:output:0:dense_9/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskї
 dense_9/ActivityRegularizer/CastCast2dense_9/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ф
#dense_9/ActivityRegularizer/truedivRealDiv4dense_9/ActivityRegularizer/PartitionedCall:output:0$dense_9/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: Ѓ
"net_output/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0net_output_51407*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_net_output_layer_call_and_return_conditional_losses_51406¤
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
GPU 2J 8ѓ *:
f5R3
1__inference_net_output_activity_regularizer_51371
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
valueB:В
,net_output/ActivityRegularizer/strided_sliceStridedSlice-net_output/ActivityRegularizer/Shape:output:0;net_output/ActivityRegularizer/strided_slice/stack:output:0=net_output/ActivityRegularizer/strided_slice/stack_1:output:0=net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskњ
#net_output/ActivityRegularizer/CastCast5net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ┤
&net_output/ActivityRegularizer/truedivRealDiv7net_output/ActivityRegularizer/PartitionedCall:output:0'net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: Ю
tf.math.subtract_9/Sub/yConst*
_output_shapes
:*
dtype0*Q
valueHBF"<    ═╠╠=Ўnя>═╠╠=*ё8>═╠╠=Ъ=>═╠╠=╝ д>═╠╠=═╠╠=═╠╠=═╠╠=    ═╠╠=Ъ
tf.math.subtract_9/SubSub+net_output/StatefulPartitionedCall:output:0!tf.math.subtract_9/Sub/y:output:0*
T0*'
_output_shapes
:         X
tf.math.pow_9/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @ё
tf.math.pow_9/PowPowtf.math.subtract_9/Sub:z:0tf.math.pow_9/Pow/y:output:0*
T0*'
_output_shapes
:         u
*tf.math.reduce_sum_9/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         Ў
tf.math.reduce_sum_9/SumSumtf.math.pow_9/Pow:z:03tf.math.reduce_sum_9/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         e
tf.math.reduce_mean_9/ConstConst*
_output_shapes
:*
dtype0*
valueB: ї
tf.math.reduce_mean_9/MeanMean!tf.math.reduce_sum_9/Sum:output:0$tf.math.reduce_mean_9/Const:output:0*
T0*
_output_shapes
: ]
tf.__operators__.add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *P7ј
tf.__operators__.add_9/AddV2AddV2#tf.math.reduce_mean_9/Mean:output:0!tf.__operators__.add_9/y:output:0*
T0*
_output_shapes
: К
add_loss_9/PartitionedCallPartitionedCall tf.__operators__.add_9/AddV2:z:0*
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
GPU 2J 8ѓ *N
fIRG
E__inference_add_loss_9_layer_call_and_return_conditional_losses_51433l
IdentityIdentity!tf.math.reduce_sum_9/Sum:output:0^NoOp*
T0*#
_output_shapes
:         g

Identity_1Identity'dense_9/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: j

Identity_2Identity*net_output/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: c

Identity_3Identity#add_loss_9/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: Ї
NoOpNoOp ^dense_9/StatefulPartitionedCall#^net_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2H
"net_output/StatefulPartitionedCall"net_output/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ћ
{
'__inference_dense_9_layer_call_fn_51798

inputs
unknown:
identityѕбStatefulPartitionedCall╩
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_51386o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
я
F
*__inference_add_loss_9_layer_call_fn_51829

inputs
identityб
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
GPU 2J 8ѓ *N
fIRG
E__inference_add_loss_9_layer_call_and_return_conditional_losses_51433O
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
і
H
1__inference_net_output_activity_regularizer_51371
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
:         G
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
Џ
~
*__inference_net_output_layer_call_fn_51814

inputs
unknown:
identityѕбStatefulPartitionedCall═
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_net_output_layer_call_and_return_conditional_losses_51406o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs"█L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*х
serving_defaultА
=
input_101
serving_default_input_10:0         D
tf.math.reduce_sum_9,
StatefulPartitionedCall:0         tensorflow/serving/predict:ЊX
ќ
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
▒

kernel
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
▒

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
Ц
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
╩
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
Ь2в
(__inference_model_24_layer_call_fn_51450
(__inference_model_24_layer_call_fn_51670
(__inference_model_24_layer_call_fn_51682
(__inference_model_24_layer_call_fn_51570└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
┌2О
C__inference_model_24_layer_call_and_return_conditional_losses_51731
C__inference_model_24_layer_call_and_return_conditional_losses_51780
C__inference_model_24_layer_call_and_return_conditional_losses_51611
C__inference_model_24_layer_call_and_return_conditional_losses_51652└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╠B╔
 __inference__wrapped_model_51345input_10"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
,
7serving_default"
signature_map
 :2dense_9/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
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
Л2╬
'__inference_dense_9_layer_call_fn_51798б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_dense_9_layer_call_and_return_all_conditional_losses_51807б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
╩
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
н2Л
*__inference_net_output_layer_call_fn_51814б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
з2­
I__inference_net_output_layer_call_and_return_all_conditional_losses_51823б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
Г
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
н2Л
*__inference_add_loss_9_layer_call_fn_51829б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_add_loss_9_layer_call_and_return_conditional_losses_51834б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
╦B╚
#__inference_signature_wrapper_51791input_10"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
▀2▄
.__inference_dense_9_activity_regularizer_51358Е
ћ▓љ
FullArgSpec
argsџ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б
	і
В2ж
B__inference_dense_9_layer_call_and_return_conditional_losses_51842б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
Р2▀
1__inference_net_output_activity_regularizer_51371Е
ћ▓љ
FullArgSpec
argsџ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б
	і
№2В
E__inference_net_output_layer_call_and_return_conditional_losses_51850б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
%:#2Adam/dense_9/kernel/m
(:&2Adam/net_output/kernel/m
%:#2Adam/dense_9/kernel/v
(:&2Adam/net_output/kernel/vЦ
 __inference__wrapped_model_51345ђ1б.
'б$
"і
input_10         
ф "GфD
B
tf.math.reduce_sum_9*і'
tf.math.reduce_sum_9         Ї
E__inference_add_loss_9_layer_call_and_return_conditional_losses_51834Dб
б
і
inputs 
ф ""б

і
0 
џ
і	
1/0 W
*__inference_add_loss_9_layer_call_fn_51829)б
б
і
inputs 
ф "і X
.__inference_dense_9_activity_regularizer_51358&б
б
і	
x
ф "і │
F__inference_dense_9_layer_call_and_return_all_conditional_losses_51807i/б,
%б"
 і
inputs         
ф "3б0
і
0         
џ
і	
1/0 А
B__inference_dense_9_layer_call_and_return_conditional_losses_51842[/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ y
'__inference_dense_9_layer_call_fn_51798N/б,
%б"
 і
inputs         
ф "і         н
C__inference_model_24_layer_call_and_return_conditional_losses_51611ї9б6
/б,
"і
input_10         
p 

 
ф "KбH
і
0         
-џ*
і	
1/0 
і	
1/1 
і	
1/2 н
C__inference_model_24_layer_call_and_return_conditional_losses_51652ї9б6
/б,
"і
input_10         
p

 
ф "KбH
і
0         
-џ*
і	
1/0 
і	
1/1 
і	
1/2 м
C__inference_model_24_layer_call_and_return_conditional_losses_51731і7б4
-б*
 і
inputs         
p 

 
ф "KбH
і
0         
-џ*
і	
1/0 
і	
1/1 
і	
1/2 м
C__inference_model_24_layer_call_and_return_conditional_losses_51780і7б4
-б*
 і
inputs         
p

 
ф "KбH
і
0         
-џ*
і	
1/0 
і	
1/1 
і	
1/2 Ђ
(__inference_model_24_layer_call_fn_51450U9б6
/б,
"і
input_10         
p 

 
ф "і         Ђ
(__inference_model_24_layer_call_fn_51570U9б6
/б,
"і
input_10         
p

 
ф "і         
(__inference_model_24_layer_call_fn_51670S7б4
-б*
 і
inputs         
p 

 
ф "і         
(__inference_model_24_layer_call_fn_51682S7б4
-б*
 і
inputs         
p

 
ф "і         [
1__inference_net_output_activity_regularizer_51371&б
б
і	
x
ф "і Х
I__inference_net_output_layer_call_and_return_all_conditional_losses_51823i/б,
%б"
 і
inputs         
ф "3б0
і
0         
џ
і	
1/0 ц
E__inference_net_output_layer_call_and_return_conditional_losses_51850[/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ |
*__inference_net_output_layer_call_fn_51814N/б,
%б"
 і
inputs         
ф "і         ┤
#__inference_signature_wrapper_51791ї=б:
б 
3ф0
.
input_10"і
input_10         "GфD
B
tf.math.reduce_sum_9*і'
tf.math.reduce_sum_9         