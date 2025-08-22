#pragma once

struct llm;

void test_fixed_llm();
void test_minimal_llm();

void log_neuron_maxes(const llm& model);