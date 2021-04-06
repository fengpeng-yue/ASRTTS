#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Optimizer module."""

import torch
import logging

class NoamOpt(object):
    """Optim wrapper that implements rate."""

    def __init__(self,lr_factor, warmup, optimizer):
        """Construct an NoamOpt object."""
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.lr_factor = lr_factor
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        """Update parameters and rate."""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above."""
        if step is None:
            step = self._step
        return (
            self.lr_factor \
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def zero_grad(self):
        """Reset gradient."""
        self.optimizer.zero_grad()

    def state_dict(self):
        """Return state_dict."""
        return {
            "_step": self._step,
            "warmup": self.warmup,
            "factor": self.factor,
            "model_size": self.model_size,
            "_rate": self._rate,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load state_dict."""
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)

class StdOpt(object):
    """Optim wrapper that implements rate."""

    def __init__(self,initial_lr,final_lr, warmup, decay_rate,decay_steps, optimizer):
        """Construct an optimizer object."""
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        #self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        """Update parameters and rate."""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above."""
        if step is None:
            step = self._step
        if step <= self.warmup:
            rate = self.initial_lr
        else:
            rate = self.initial_lr* \
                (self.decay_rate ** ((step - self.warmup) / self.decay_steps))
            rate = max(rate, self.final_lr)
        # logging.info("current setp : %d" % step)
        # logging.info("current learing rate : %f" % rate)
        return rate
        
    def zero_grad(self):
        """Reset gradient."""
        self.optimizer.zero_grad()

    def state_dict(self):
        """Return state_dict."""
        return {
            "_step": self._step,
            "initial_lr": self.initial_lr,
            "final_lr": self.final_lr,
            "warmup": self.warmup,
            "decay_rate": self.decay_rate,
            "decay_steps": self.decay_steps,     
            "_rate": self._rate,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load state_dict."""
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)


def get_std_opt(model_params,initial_lr,final_lr,eps,warmup,decay_rate,decay_steps):
    """Get standard opt."""
    base = torch.optim.Adam(model_params, lr=initial_lr, eps=eps)
    return StdOpt(initial_lr,final_lr, warmup, decay_rate,decay_steps, base)


def get_noam_opt(model_params,warmup,lr_factor):
    """Get standard NoamOpt."""
    base = torch.optim.Adam(model_params, lr=0, betas=(0.9, 0.98), eps=1e-9)
    return NoamOpt(lr_factor, warmup, base)

