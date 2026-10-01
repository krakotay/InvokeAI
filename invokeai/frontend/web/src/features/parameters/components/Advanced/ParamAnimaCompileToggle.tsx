import { Flex, FormControl, FormHelperText, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectAnimaCompileBlocks, setAnimaCompileBlocks } from 'features/controlLayers/store/paramsSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const ParamAnimaCompileToggle = memo(() => {
  const isChecked = useAppSelector(selectAnimaCompileBlocks);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const onChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => dispatch(setAnimaCompileBlocks(event.target.checked)),
    [dispatch]
  );

  return (
    <FormControl orientation="vertical" gap={1}>
      <Flex w="full" alignItems="center" justifyContent="space-between" gap={2}>
        <FormLabel m={0} htmlFor="anima-compile-blocks">
          {t('parameters.animaCompileBlocks')}
        </FormLabel>
        <Switch id="anima-compile-blocks" isChecked={isChecked} onChange={onChange} />
      </Flex>
      <FormHelperText m={0}>{t('parameters.animaCompileBlocksHelp')}</FormHelperText>
    </FormControl>
  );
});

ParamAnimaCompileToggle.displayName = 'ParamAnimaCompileToggle';
