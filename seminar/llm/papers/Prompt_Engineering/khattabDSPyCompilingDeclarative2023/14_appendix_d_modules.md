# D Modules

# D. 1 Predict
```
class Predict(dspy.Module):
    def __init__(self, signature, **config):
        self.signature = dspy.Signature(signature)
        self.config = config
        # Module Parameters.
        self.lm = dspy.ParameterLM(None) # use the default LM
        self.demonstrations = dspy.ParameterDemonstrations([])
    def forward(self, **kwargs):
        lm = get_the_right_lm(self.lm, kwargs)
        signature = get_the_right_signature(self.signature, kwargs)
        demonstrations = get_the_right_demonstrations(self.demonstrations, kwargs)
        prompt = signature(demos=self.demos, **kwargs)
        completions = lm.generate(prompt, **self.config)
        prediction = Prediction.from_completions(completions, signature=signature)
        if dsp.settings.compiling is not None:
            trace = dict(predictor=self, inputs=kwargs, outputs=prediction)
            dspy.settings.traces.append(trace)
        return prediction
```

# D. 2 Chain of Thought
```
class ChainOfThought(dspy.Module):
    def __init__(self, signature):
        # Modify signature from '*inputs -> *outputs' to '*inputs -> rationale, *outputs'.
        rationale_field = dspy.OutputField(prefix="Reasoning: Let’s think step by step.")
        signature = dspy. Signature(signature).prepend_output_field(rationale_field)
        # Declare a sub-module with the modified signature.
        self.predict = dspy.Predict(self.signature)
    def forward(self, **kwargs):
        # Just forward the inputs to the sub-module.
        return self.predict(**kwargs)
```

