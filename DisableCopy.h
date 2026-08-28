#pragma once
#include <iostream>
struct DisableCopy
{
	DisableCopy() = default;
	DisableCopy(DisableCopy const&) = delete;
	DisableCopy& operator=(DisableCopy const&) = delete;
};