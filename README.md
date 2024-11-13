# Instructions MapReduce
Similar with Hadoop MapReduce, but this time we map "instructions" into LLM's responses, and then reduce these response to the final results or next Map/Reduce's inputs.

## Architecture
```mermaid
graph TD
Input[Input]
Input --fields scope 1-->InputSubet1[Mapping Inputs 1]
Input --fields scope 2-->InputSubet2[Mapping Inputs 2]
Input --fields scope n-->InputSubetN[Mapping Inputs n]
InputSubet1 --> Instruction1(Map Instruction 1)
InputSubet2 --> Instruction2(Map Instruction 2)
InputSubetN --> InstructionN(Map Instruction n)
Instruction1 --> LlmMapper(LLM Based Mapper)
Instruction2 --> LlmMapper
InstructionN --> LlmMapper 
LlmMapper --> MappingOutput1[Mapping Output 1]
LlmMapper --> MappingOutput2[Mapping Output 2]
LlmMapper --> MappingOutputN[Mapping Output n]
MappingOutput1 --> MappingOutputs[Structured Mapping Output]
MappingOutput2 --> MappingOutputs 
MappingOutputN --> MappingOutputs
MappingOutputs --fields scope 1--> ReduceInputs1(Reduce Inputs 1)
MappingOutputs --fields scope 2--> ReduceInputs2(Reduce Inputs 2)
MappingOutputs --fields scope m--> ReduceInputsM(Reduce Inputs m)
ReduceInputs1 --> ReduceInstruction1(Reduce Instruction 1)
ReduceInputs2 --> ReduceInstruction2(Reduce Instruction 2)
ReduceInputsM --> ReduceInstructionM(Reduce Instruction m)
ReduceInstruction1 --> LlmReducer(Llm Based Reducer)
ReduceInstruction2 --> LlmReducer 
ReduceInstructionM --> LlmReducer 
LlmReducer --> ReduceOutput1(Reduce Output 1)
LlmReducer --> ReduceOutput2(Reduce Output 2)
LlmReducer --> ReduceOutputM(Reduce Output M)
```

