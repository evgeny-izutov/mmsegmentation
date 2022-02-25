# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from logging import Logger

import pytest
from mmseg.apis.ote.extension.utils.hooks import (
    CancelTrainingHook,
    EnsureCorrectBestCheckpointHook,
    FixedMomentumUpdaterHook,
    OTELoggerHook,
    OTEProgressHook,
)

from ote_sdk.test_suite.e2e_test_system import e2e_pytest_unit
from ote_sdk.tests.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)
from ote_sdk.usecases.reporting.time_monitor_callback import TimeMonitorCallback


class TestCancelTrainingHook:
    @e2e_pytest_unit
    def test_cancel_training_hook_init_params_validation(self):
        """
        <b>Description:</b>
        Check CancelTrainingHook object initialization parameters validation

        <b>Input data:</b>
        "interval" non-int type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        CancelTrainingHook object initialization parameter
        """
        with pytest.raises(ValueError):
            CancelTrainingHook(interval="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_cancel_training_hook_after_train_iter_params_validation(self):
        """
        <b>Description:</b>
        Check CancelTrainingHook object "after_train_iter" method input parameters validation

        <b>Input data:</b>
        CancelTrainingHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "after_train_iter" method
        """
        hook = CancelTrainingHook()
        with pytest.raises(ValueError):
            hook.after_train_iter(runner="unexpected string")  # type: ignore


class TestFixedMomentumUpdaterHook:
    @e2e_pytest_unit
    def test_fixed_momentum_updater_hook_before_run_params_validation(self):
        """
        <b>Description:</b>
        Check FixedMomentumUpdaterHook object "before_run" method input parameters validation

        <b>Input data:</b>
        FixedMomentumUpdaterHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "before_run" method
        """
        hook = FixedMomentumUpdaterHook()
        with pytest.raises(ValueError):
            hook.before_run(runner="unexpected string")  # type: ignore


class TestEnsureCorrectBestCheckpointHook:
    @e2e_pytest_unit
    def test_ensure_correct_best_checkpoint_hook_after_run_params_validation(self):
        """
        <b>Description:</b>
        Check EnsureCorrectBestCheckpointHook object "after_run" method input parameters validation

        <b>Input data:</b>
        EnsureCorrectBestCheckpointHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "after_run" method
        """
        hook = EnsureCorrectBestCheckpointHook()
        with pytest.raises(ValueError):
            hook.after_run(runner="unexpected string")  # type: ignore


class TestOTELoggerHook:
    @e2e_pytest_unit
    def test_ote_logger_hook_init_params_validation(self):
        """
        <b>Description:</b>
        Check OTELoggerHook object initialization parameters validation

        <b>Input data:</b>
        OTELoggerHook object initialization parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        OTELoggerHook object initialization parameter
        """
        correct_values_dict = {}
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "curves" parameter
            ("curves", unexpected_str),
            # Unexpected string is specified as nested curve
            (
                "curves",
                {
                    "expected": OTELoggerHook.Curve(),
                    "unexpected": unexpected_str,
                },
            ),
            # Unexpected string is specified as "interval" parameter
            ("interval", unexpected_str),
            # Unexpected string is specified as "ignore_last" parameter
            ("ignore_last", unexpected_str),
            # Unexpected string is specified as "reset_flag" parameter
            ("reset_flag", unexpected_str),
            # Unexpected string is specified as "by_epoch" parameter
            ("by_epoch", unexpected_str),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=OTELoggerHook,
        )

    @e2e_pytest_unit
    def test_ote_logger_hook_log_params_validation(self):
        """
        <b>Description:</b>
        Check OTELoggerHook object "log" method input parameters validation

        <b>Input data:</b>
        OTELoggerHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "log" method
        """
        hook = OTELoggerHook()
        with pytest.raises(ValueError):
            hook.log(runner="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_ote_logger_hook_after_train_epoch_params_validation(self):
        """
        <b>Description:</b>
        Check OTELoggerHook object "after_train_epoch" method input parameters validation

        <b>Input data:</b>
        OTELoggerHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "after_train_epoch" method
        """
        hook = OTELoggerHook()
        with pytest.raises(ValueError):
            hook.after_train_epoch(runner="unexpected string")  # type: ignore


class TestOTEProgressHook:
    @staticmethod
    def time_monitor():
        return TimeMonitorCallback(
            num_epoch=10, num_train_steps=5, num_test_steps=5, num_val_steps=4
        )

    def hook(self):
        return OTEProgressHook(time_monitor=self.time_monitor())

    @e2e_pytest_unit
    def test_ote_progress_hook_init_params_validation(self):
        """
        <b>Description:</b>
        Check OTEProgressHook object initialization parameters validation

        <b>Input data:</b>
        OTEProgressHook object initialization parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        OTEProgressHook object initialization parameter
        """
        correct_values_dict = {"time_monitor": self.time_monitor()}
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "time_monitor" parameter
            ("time_monitor", unexpected_str),
            # Unexpected string is specified as "verbose" parameter
            ("verbose", unexpected_str),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=OTEProgressHook,
        )

    @e2e_pytest_unit
    def test_ote_progress_hook_before_run_params_validation(self):
        """
        <b>Description:</b>
        Check OTEProgressHook object "before_run" method input parameters validation

        <b>Input data:</b>
        OTEProgressHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "before_run" method
        """
        hook = self.hook()
        with pytest.raises(ValueError):
            hook.before_run(runner="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_ote_progress_hook_before_epoch_params_validation(self):
        """
        <b>Description:</b>
        Check OTEProgressHook object "before_epoch" method input parameters validation

        <b>Input data:</b>
        OTEProgressHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "before_epoch" method
        """
        hook = self.hook()
        with pytest.raises(ValueError):
            hook.before_epoch(runner="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_ote_progress_hook_after_epoch_params_validation(self):
        """
        <b>Description:</b>
        Check OTEProgressHook object "after_epoch" method input parameters validation

        <b>Input data:</b>
        OTEProgressHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "after_epoch" method
        """
        hook = self.hook()
        with pytest.raises(ValueError):
            hook.after_epoch(runner="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_ote_progress_hook_before_iter_params_validation(self):
        """
        <b>Description:</b>
        Check OTEProgressHook object "before_iter" method input parameters validation

        <b>Input data:</b>
        OTEProgressHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "before_iter" method
        """
        hook = self.hook()
        with pytest.raises(ValueError):
            hook.before_iter(runner="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_ote_progress_hook_after_iter_params_validation(self):
        """
        <b>Description:</b>
        Check OTEProgressHook object "after_iter" method input parameters validation

        <b>Input data:</b>
        OTEProgressHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "after_iter" method
        """
        hook = self.hook()
        with pytest.raises(ValueError):
            hook.after_iter(runner="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_ote_progress_hook_before_val_iter_params_validation(self):
        """
        <b>Description:</b>
        Check OTEProgressHook object "before_val_iter" method input parameters validation

        <b>Input data:</b>
        OTEProgressHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "before_val_iter" method
        """
        hook = self.hook()
        with pytest.raises(ValueError):
            hook.before_val_iter(runner="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_ote_progress_hook_after_val_iter_params_validation(self):
        """
        <b>Description:</b>
        Check OTEProgressHook object "after_val_iter" method input parameters validation

        <b>Input data:</b>
        OTEProgressHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "after_val_iter" method
        """
        hook = self.hook()
        with pytest.raises(ValueError):
            hook.after_val_iter(runner="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_ote_progress_hook_after_run_params_validation(self):
        """
        <b>Description:</b>
        Check OTEProgressHook object "after_run" method input parameters validation

        <b>Input data:</b>
        OTEProgressHook object, "runner" non-BaseRunner type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "after_run" method
        """
        hook = self.hook()
        with pytest.raises(ValueError):
            hook.after_run(runner="unexpected string")  # type: ignore
