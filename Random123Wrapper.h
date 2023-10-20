#pragma once

#include "Random123/philox.h"
typedef r123::Philox4x64 RNG;

class Random123Wrapper {
public:
  Random123Wrapper();

  void SetStep(ulong step);
  void SetRandomSeed(ulong seedValue);
  void SetKey(unsigned int key);

  unsigned int GetStep() const;
  unsigned int GetKeyValue() const;
  unsigned int GetSeedValue() const;
  double operator()(unsigned int counter);

  double GetRandomNumber(unsigned int counter);
  double GetSymRandom(unsigned int counter, double bound);
  double GetGaussian(unsigned int counter);
  double GetGaussianNumber(unsigned int counter, double mean, double stdDev);

private:
  inline RNG::ctr_type getRNG(unsigned int counter);

  RNG::ctr_type c;
  RNG::key_type uk;
  RNG rng;
};

#include "Random123/boxmuller.hpp"
#include "Random123/uniform.hpp"

Random123Wrapper::Random123Wrapper() {
  c = {{}};
  uk = {{}};
}

inline RNG::ctr_type Random123Wrapper::getRNG(unsigned int counter) {
  // Need to use the localc variable to avoid OpenMP race conditions
  RNG::ctr_type localc = {{}};
  localc[0] = counter;
  localc[1] = GetKeyValue();
  return rng(localc, uk);
}

void Random123Wrapper::SetStep(ulong step) { uk[0] = step; }

void Random123Wrapper::SetRandomSeed(ulong seedValue) { uk[1] = seedValue; }

void Random123Wrapper::SetKey(unsigned int key) { c[1] = key; }

unsigned int Random123Wrapper::GetStep() const { return uk[0]; }

unsigned int Random123Wrapper::GetKeyValue() const { return c[1]; }

unsigned int Random123Wrapper::GetSeedValue() const { return uk[1]; }

double Random123Wrapper::operator()(unsigned int counter) {
  return GetRandomNumber(counter);
}

double Random123Wrapper::GetRandomNumber(unsigned int counter) {
  RNG::ctr_type r = getRNG(counter);
  double r01 = r123::u01<double>(r[0]);
  return r01;
}

double Random123Wrapper::GetSymRandom(unsigned int counter, double bound) {
  RNG::ctr_type r = getRNG(counter);
  double r01;
  r01 = bound * r123::uneg11<double>(r[0]);
  return r01;
}

double Random123Wrapper::GetGaussian(unsigned int counter) {
  RNG::ctr_type r = getRNG(counter);
  r123::double2 normal2 = r123::boxmuller(r[0], r[1]);
  return normal2.x;
}

double Random123Wrapper::GetGaussianNumber(unsigned int counter, double mean,
                                           double stdDev) {
  double gNum = this->GetGaussian(counter);
  return (mean + gNum * stdDev);
}