# E Teleprompters

# E. 1 BootstrapFewShot
```
class SimplifiedBootstrapFewShot(Teleprompter):
    def __init__(self, metric=None):
        self.metric = metric
    def compile(self, student, trainset, teacher=None):
        teacher = teacher if teacher is not None else student
        compiled_program = student.deepcopy()
        # Step 1. Prepare mappings between student and teacher Predict modules.
        # Note: other modules will rely on Predict internally.
        assert student_and_teacher_have_compatible_predict_modules(student, teacher)
        name2predictor, predictor2name = map_predictors_recursively(student, teacher)
        # Step 2. Bootstrap traces for each Predict module.
        # We'll loop over the training set. We'll try each example once for simplicity.
        for example in trainset:
            if we_found_enough_bootstrapped_demos(): break
            # turn on compiling mode which will allow us to keep track of the traces
            with dspy.setting.context(compiling=True):
                # run the teacher program on the example, and get its final prediction
                # note that compiling=True may affect the internal behavior here
                prediction = teacher(**example.inputs())
                # get the trace of the all interal Predict calls from teacher program
                predicted_traces = dspy.settings.trace
            # if the prediction is valid, add the example to the traces
            if self.metric(example, prediction, predicted_traces):
                for predictor, inputs, outputs in predicted_traces:
                    d = dspy.Example(automated=True, **inputs, **outputs)
                    predictor_name = self.predictor2name[id(predictor)]
                    compiled_program[predictor_name].demonstrations.append(d)
        return compiled_program
```

# E. 2 BootstrapFewShotWithRandomSearch
```
class SimplifiedBootstrapFewShotWithRandomSearch(Teleprompter):
    def __init__(self, metric = None, trials=16):
        self.metric = metric
        self.trials = trials
    def compile(self, student, *, teacher=None, trainset, valset=None):
        # we can do forms of cross-validation if valset is unset.
        valset = trainset if valset is None else valset
        candidates = []
        for seed in range(self.trials):
            # Create a new basic bootstrap few-shot program.
            shuffled_trainset = shuffle(trainset, seed=seed)
            tp = BootstrapFewShot(metric=metric, max_bootstrap_demos=random_size())
            candidate_program = tp.compile(student, shuffled_trainset, teacher)
            # Step 2: Evaluate the generated candidate program.
            score = evaluate_program(candidate_program, self.metric, valset)
            candidates.append((score, candidate_program))
        # return the best candidate program.
        return max(candidates, key=lambda x: x[0])[1]
```

# E. 3 BootstrapFewShotWithOptuna
```
class SimplifiedBootstrapFewShotWithOptuna(Teleprompter):
    def __init__(self, metric, trials=16):
        self.metric = metric
        self.trials = trials
    def objective(self, trial):
        pool = self.pool
        # Step 1: Create copy of student program.
        candidate_program = self.student.reset_copy()
        # Step 2: Based on trial, select demos for each predictor in program.
        # Note. For simplicity, we can just select a single demo for each predictor.
        # But we can easily tune the number of demonstrations to select here.
        for (name, predictor1), (_, predictor2) in \
        zip(pool.named_predictors(), candidate_program.named_predictors()):
            all_demos = predictor1.demos
            demo_index = trial.suggest_int(f"demo_index_for_{name}", 0, len(all_demos) - 1)
            predictor2.demos = [all_demos[demo_index]]
        # Step 3: Evaluate the modified candidate program.
        score = evaluate_program(candidate_program, self.metric, self.valset)
        # Step 4: Store the candidate for Optuna to select highest-scoring program.
        trial.set_user_attr("program", candidate_program)
        return score
    def compile(self, student, trainset, teacher=None, valset=None):
        self.trainset = trainset
        self.valset = trainset if valset is None else valset
        self.student = student.deepcopy()
        self.teacher = teacher.deepcopy() if teacher else student.deepcopy()
        # Leverage BootstrapFewshot to create a large number of potential demonstrations.
        tp = BootstrapFewShot()
        self.pool = tp.compile(self.student, self.teacher, self.trainset, self.metric)
        # Use Optuna to find the best program by optimizing the objective function.
        best_program = optimize_with_optuna(self.objective)
        print(’Best score:’, best_program.score)
        print(’Best program:’, best_program)
        return best_program
```

