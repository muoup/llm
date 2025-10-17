#pragma once

#include <functional>

struct dataset {
    virtual ~dataset() = default;
  
    virtual void for_each(std::function<void(std::string_view)>) const = 0;
};

struct row_dataset : public dataset {
    std::string data;
    std::vector<std::string_view> rows;
    
    void for_each(std::function<void(std::string_view)> func) const override {
        for (const auto& row : rows) {
            func(row);
        }
    };
};

struct raw_dataset : public dataset {
    std::string data;
    
    void for_each(std::function<void(std::string_view)> func) const override {
        func(data);
    }
};