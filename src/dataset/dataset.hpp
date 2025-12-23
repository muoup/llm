#pragma once

#include <functional>
#include <string_view>
#include <string>

struct dataset {
    virtual ~dataset() = default;
  
    virtual size_t size() const = 0;
    virtual void for_each(std::function<void(std::string_view)>) const = 0;
    void enumerate(std::function<void(size_t, std::string_view)> func) const {
        size_t index = 0;
        for_each([&](std::string_view row) {
            func(index++, row);
        });
    }
};

struct row_dataset : public dataset {
    std::string data;
    std::vector<std::string_view> rows;
    
    size_t size() const override {
        return rows.size();
    }
    void for_each(std::function<void(std::string_view)> func) const override {
        for (const auto& row : rows) {
            func(row);
        }
    };
};

struct raw_dataset : public dataset {
    std::string data;
    
    size_t size() const override {
        return 1;
    }
    void for_each(std::function<void(std::string_view)> func) const override {
        func(data);
    }
};

// For use with ensuring that a model's training loop works correctly.
// Takes the first 250 characters of data and functions as if it is a dataset
// of that same data repeated multiple times, allowing for the model to overfit
// on that small sample.
struct overfit_dataset : public dataset {
    std::string data;
    size_t repeat_count = 1000;

    size_t size() const override {
        return repeat_count;
    }
    void for_each(std::function<void(std::string_view)> func) const override {
        for (size_t i = 0; i < repeat_count; ++i) {
            func(data);
        }
    }
};